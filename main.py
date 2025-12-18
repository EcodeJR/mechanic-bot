import os
import json
import requests
import traceback
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from groq import Groq 
from google import genai 
from google.genai import types 
import uvicorn
from fastapi.responses import PlainTextResponse

load_dotenv()
app = FastAPI()

# --- 1. INITIALIZATION ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Gemini Client initialized.")
except Exception as e:
    print(f"❌ ERROR: Gemini Init failed: {e}")
    gemini_client = None 

USER_SESSIONS = {}
INVENTORY = []

def load_inventory():
    global INVENTORY
    try:
        # Try MERN Backend (Update this URL after hosting on Vercel)
        
        api_url = os.getenv("API_URL", "https://inventory-mng-backend.vercel.app/api/inventory")
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            INVENTORY = response.json()
            print(f"✅ Loaded {len(INVENTORY)} items from MERN.")
            return
    except Exception as e:
        print(f"⚠️ MERN API Offline: {e}")

    try:
        with open('inventory.json', 'r') as f:
            INVENTORY = json.load(f)
        print(f"📦 Loaded {len(INVENTORY)} items from local JSON.")
    except Exception as e:
        print(f"❌ Critical: No inventory found.")
        INVENTORY = []

load_inventory()

def calculate_match_score(query_str, target_str):
    if not query_str or not target_str: return 0
    query_words = set(query_str.lower().split())
    target_words = set(target_str.lower().split())
    common = query_words.intersection(target_words)
    if not common: return 0
    # Returns percentage of overlap
    return (len(common) / len(query_words)) * 100

def send_whatsapp_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v19.0/{os.getenv('WHATSAPP_PHONE_ID')}/messages"
    headers = {"Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message_text},
    }
    requests.post(url, headers=headers, json=data)

# --- 3. ENDPOINTS ---
@app.get("/")
async def home():
    return {
        "status": "Mechanic Bot Online", 
        "inventory_count": len(INVENTORY),
        "database_status": "Connected" if INVENTORY else "Empty/Offline"
    }
    
@app.get("/webhook")
async def verify_webhook(request: Request):
    # This must match exactly what you have in Render/Meta
    token = os.getenv("VERIFY_TOKEN") 
    
    query = request.query_params
    mode = query.get("hub.mode")
    verify_token = query.get("hub.verify_token")
    challenge = query.get("hub.challenge")
    
    if mode == "subscribe" and verify_token == token:
        print("✅ Webhook Verified Successfully!")
        return PlainTextResponse(content=challenge) # Must be plain text
    
    print("❌ Verification Failed: Token Mismatch")
    raise HTTPException(status_code=403, detail="Verification failed")
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    try: 
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" not in entry: return {"status": "ok"}
        
        message = entry["messages"][0]
        sender_id = message["from"]
        
        # --- IMAGE PROCESSING ---
        if message.get("type") == "image":
            image_id = message["image"]["id"]
            headers = {"Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}"}
            media_url = requests.get(f"https://graph.facebook.com/v19.0/{image_id}", headers=headers).json()["url"]
            image_bytes = requests.get(media_url, headers=headers).content
            
            # 1. Vision with Strict Prompting (Fixed Model String)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
            vision_res = gemini_client.models.generate_content(
                model='gemini-robotics-er-1.5-preview', # Added 'models/' prefix to fix 404
                contents=[image_part, "Identify the car part. Format: 'PART: [name] | MODEL: [car]'. If model unknown, say 'MODEL: Unknown'. Be extremely concise."]
            )
            description = vision_res.text.strip()
            print(f"AI Vision Raw: {description}")

            # 2. Llama Cleaning (Strict instruction to avoid explanations)
            llama_res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Extract ONLY 'PartName - CarModel' from: '{description}'. No sentences. No explanations. Example: 'Brake Pads - Lexus RX350'."}]
            )
            extracted_text = llama_res.choices[0].message.content.strip()
            print(f"Extracted: {extracted_text}")

            try: search_part, search_model = extracted_text.lower().split(' - ', 1)
            except: search_part, search_model = extracted_text.lower(), "unknown"

            # 3. Search Logic
            best_match = None
            highest_score = 0
            for item in INVENTORY:
                p_score = calculate_match_score(search_part, item['part_name'])
                m_score = calculate_match_score(search_model, item['vehicle']) if "unknown" not in search_model else 0
                
                total = (p_score * 0.7) + (m_score * 0.3)
                if total > highest_score:
                    highest_score = total
                    best_match = item

            # 4. Final Response Logic
            if highest_score > 65: 
                reply = f"✅ Found: {best_match['part_name']}\n🚙 Vehicle: {best_match['vehicle']}\n💰 Price: N{best_match['price_NGN']:,}\n📦 Stock: {best_match['stock_qty']}"
            elif "unknown" in search_model or highest_score < 40:
                USER_SESSIONS[sender_id] = {"part_name": search_part}
                reply = f"🔧 I've identified this as a **{search_part.title()}**.\n\nWhich **Car Model and Year** do you need this for?"
            else:
                reply = f"🔍 I identified a {search_part}, but no exact match for {search_model} in inventory."

            send_whatsapp_message(sender_id, reply)

        # --- TEXT PROCESSING ---
        elif message.get("type") == "text":
            text_body = message["text"]["body"].lower()
            
            if sender_id in USER_SESSIONS:
                saved_part = USER_SESSIONS.pop(sender_id)["part_name"]
                search_part, search_model = saved_part, text_body
            else:
                # Fixed Llama prompt to stop it from writing essays
                llama_res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"Extract ONLY 'PartName - CarModel' from: '{text_body}'. No extra text. Output Example: 'Headlamp - Toyota Corolla'."}]
                )
                extracted_text = llama_res.choices[0].message.content.strip().lower()
                try: search_part, search_model = extracted_text.split(' - ', 1)
                except: search_part, search_model = extracted_text, "unknown"

            # --- FUZZY TEXT SEARCH ---
            best_match = None
            highest_score = 0
            for item in INVENTORY:
                p_score = calculate_match_score(search_part, item['part_name'])
                m_score = calculate_match_score(search_model, item['vehicle']) if "unknown" not in search_model else 0
                
                total = (p_score * 0.7) + (m_score * 0.3)
                if total > highest_score:
                    highest_score = total
                    best_match = item
            
            if highest_score > 60:
                reply = f"✅ Found: {best_match['part_name']} ({best_match['vehicle']})\n💰 Price: N{best_match['price_NGN']:,} \n 📌 Location: {best_match['location']}"
            else:
                reply = f"Sorry, I couldn't find a {search_part} for {search_model}. We may not have it in stock currently, check back in a few days or Please double check the model name(You may have sent the wrong model or year.)."
                
            send_whatsapp_message(sender_id, reply)

    except Exception:
        traceback.print_exc()

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)