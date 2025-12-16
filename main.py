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

# Load passwords and tokens from .env file
load_dotenv()

app = FastAPI()

# --- 1. INITIALIZATION ---

# Groq Client for Llama Intelligence
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini Client for Vision
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Gemini Client initialized.")
except Exception as e:
    print(f"❌ ERROR: Could not initialize Gemini Client: {e}")
    gemini_client = None 

# --- GLOBAL SESSION MEMORY ---
USER_SESSIONS = {}

# --- INVENTORY MANAGEMENT ---
INVENTORY = []

def load_inventory():
    """Loads inventory: Tries MERN API first, falls back to local JSON."""
    global INVENTORY
    
    # Step 1: Try to get data from your MERN Backend (Live Data)
    try:
        api_url = "http://localhost:5000/api/products"
        response = requests.get(api_url, timeout=5) # Added timeout to prevent hanging
        if response.status_code == 200:
            INVENTORY = response.json()
            print(f"✅ SUCCESS: Loaded {len(INVENTORY)} items from MERN Database.")
            return # Exit function if API works
        else:
            print(f"⚠️ MERN API returned status {response.status_code}. Falling back to local file.")
    except Exception as e:
        print(f"⚠️ Could not connect to MERN API: {e}. Falling back to local file.")

    # Step 2: Fallback to Local JSON (If API fails)
    try:
        with open('inventory.json', 'r') as f:
            INVENTORY = json.load(f)
        print(f"📦 LOCAL FALLBACK: Loaded {len(INVENTORY)} items from inventory.json.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load local inventory.json either: {e}")
        INVENTORY = []

# Initial load on startup
load_inventory()

def calculate_match_score(query_str, target_str):
    """Returns a score (0-100) based on word overlap."""
    if not query_str or not target_str:
        return 0
    query_words = set(query_str.lower().split())
    target_words = set(target_str.lower().split())
    common_words = query_words.intersection(target_words)
    if not common_words: return 0
    score_v1 = len(common_words) / len(query_words)
    score_v2 = len(common_words) / len(target_words)
    return max(score_v1, score_v2) * 100

# --- 2. WHATSAPP UTILITY ---

def send_whatsapp_message(recipient_id, message_text):
    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN") 
    PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID") 
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("❌ ERROR: WhatsApp Credentials missing.")
        return
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
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
    return {"status": "Mechanic Bot Online", "inventory_count": len(INVENTORY)}

@app.get("/webhook")
async def verify_webhook(request: Request):
    token = os.getenv("VERIFY_TOKEN")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if verify_token == token:
        return int(challenge)
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
            try:
                image_id = message["image"]["id"]
                headers = {"Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}"}
                media_url = requests.get(f"https://graph.facebook.com/v19.0/{image_id}", headers=headers).json()["url"]
                image_bytes = requests.get(media_url, headers=headers).content
                
                # Vision (Gemini)
                image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                vision_res = gemini_client.models.generate_content(
                    model='gemini-2.0-flash-lite',
                    contents=[image_part, "Describe this car part in detail, include type/model/year. Concise."]
                )
                description = vision_res.text.strip()

                # Llama Extraction
                llama_res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"Extract 'Part Name - Car Model' from: '{description}'. If unknown model, use 'Unknown Model'."}]
                )
                extracted_text = llama_res.choices[0].message.content.strip()

                try: search_part, search_model = extracted_text.lower().split(' - ', 1)
                except: search_part, search_model = extracted_text.lower(), ""

                # Smart Search
                best_match = None
                highest_score = 0
                for item in INVENTORY:
                    p_score = calculate_match_score(search_part, item['part_name'])
                    m_score = calculate_match_score(search_model, item['vehicle']) if search_model and "unknown" not in search_model else 100
                    if m_score < 30: continue
                    total = p_score + m_score
                    if total > highest_score and total > 80:
                        highest_score = total
                        best_match = item
                
                found_part = best_match

                # AI Verification
                if found_part:
                    verify_prompt = f"User wants: '{search_part}' for '{search_model}'. Inventory match: '{found_part['part_name']}' for '{found_part['vehicle']}'. Reasonable? YES/NO."
                    verify_res = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": verify_prompt}]
                    )
                    if "YES" not in verify_res.choices[0].message.content.upper():
                        found_part = None

                # Reply Logic
                if found_part:
                    reply = f"✅ Found: {found_part['part_name']}\nVehicle: {found_part['vehicle']}\nPrice: N{found_part['price_NGN']:,}\nStock: {found_part['stock_qty']}"
                elif "unknown model" in extracted_text.lower():
                    USER_SESSIONS[sender_id] = {"step": "waiting_for_model", "part_name": search_part}
                    reply = f"🔍 Identified {search_part}, but I need the Car Model. Please reply with the Model & Year."
                else:
                    reply = "🔍 Identified the part, but no exact match in inventory."

                send_whatsapp_message(sender_id, reply)
            except Exception as e:
                traceback.print_exc()
                send_whatsapp_message(sender_id, "Error processing image.")

        # --- TEXT PROCESSING ---
        elif message.get("type") == "text":
            text_body = message["text"]["body"]
            
            if sender_id in USER_SESSIONS:
                saved_part = USER_SESSIONS.pop(sender_id)["part_name"]
                prompt = f"User searching for '{saved_part}'. They said '{text_body}'. Extract Car Model. Output: '{saved_part} - Car Model'."
            else:
                prompt = f"Extract 'Part Name - Car Model' from: '{text_body}'."

            try:
                llama_res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                extracted_text = llama_res.choices[0].message.content.strip().split('\n')[0]
                
                try: search_part, search_model = extracted_text.lower().split(' - ', 1)
                except: search_part, search_model = extracted_text.lower(), ""

                found_part = None
                for item in INVENTORY:
                    if search_part in item['part_name'].lower():
                        if not search_model or search_model in item['vehicle'].lower():
                            found_part = item
                            break
                
                if found_part:
                    reply = f"✅ Found: {found_part['part_name']} ({found_part['vehicle']})\nPrice: N{found_part['price_NGN']:,}"
                else:
                    reply = f"Could not find exact match for {extracted_text}. Try sending a photo."
                
                send_whatsapp_message(sender_id, reply)
            except Exception as e:
                send_whatsapp_message(sender_id, "Error processing text.")

    except Exception as e:
        print(f"Parse Error: {e}")

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)