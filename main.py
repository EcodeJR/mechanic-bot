import os
import json
import requests
import traceback
from typing import Any, Dict, List, Optional
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
BACKEND_API_URL = os.getenv("PARTFINDR_API_URL", "http://localhost:5000/api/v1")
BOT_SECRET = os.getenv("PARTFINDR_BOT_SECRET", os.getenv("WHATSAPP_TOKEN", ""))


def backend_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Bot-Secret": BOT_SECRET,
    }

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


def local_best_match(search_part: str, search_model: str) -> tuple[Optional[Dict[str, Any]], float]:
    best_match = None
    highest_score = 0.0

    for item in INVENTORY:
        p_score = calculate_match_score(search_part, item.get('part_name', ''))
        m_score = calculate_match_score(search_model, item.get('vehicle', '')) if "unknown" not in search_model else 0
        total = (p_score * 0.7) + (m_score * 0.3)
        if total > highest_score:
            highest_score = total
            best_match = item

    return best_match, highest_score


def search_partfindr_backend(part_name: str, category: str = "") -> List[Dict[str, Any]]:
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/bot/search",
            headers=backend_headers(),
            json={"partName": part_name, "category": category or None},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", []) if isinstance(payload, dict) else []
    except Exception as err:
        print(f"⚠️ Backend search unavailable, using local fallback: {err}")
        return []


def log_partfindr_session(payload: Dict[str, Any]) -> None:
    try:
        requests.post(
            f"{BACKEND_API_URL}/bot/sessions",
            headers=backend_headers(),
            json=payload,
            timeout=5,
        )
    except Exception as err:
        print(f"⚠️ Failed to log backend session: {err}")


def format_backend_result(results: List[Dict[str, Any]]) -> str:
    top = results[:3]
    lines = ["✅ Found matching listings:"]
    for idx, item in enumerate(top, start=1):
        price = item.get("price", 0)
        lines.append(
            f"{idx}. {item.get('partName', 'Part')} | {item.get('sellerName', 'Seller')} | N{price:,}\n"
            f"   ⭐ {item.get('sellerRating', 0)} | {item.get('sellerLocation', 'Unknown')}"
        )

    return "\n".join(lines)

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

            backend_results = search_partfindr_backend(f"{search_part} {search_model}".strip())

            # 3. Final Response Logic
            if backend_results:
                reply = format_backend_result(backend_results)
            else:
                best_match, highest_score = local_best_match(search_part, search_model)
                if best_match and highest_score > 65:
                    reply = (
                        f"✅ Found: {best_match['part_name']}\n"
                        f"🚙 Vehicle: {best_match['vehicle']}\n"
                        f"💰 Price: N{best_match['price_NGN']:,}\n"
                        f"📦 Stock: {best_match['stock_qty']}"
                    )
                elif "unknown" in search_model or highest_score < 40:
                    USER_SESSIONS[sender_id] = {"part_name": search_part}
                    reply = f"🔧 I've identified this as a **{search_part.title()}**.\n\nWhich **Car Model and Year** do you need this for?"
                else:
                    reply = f"🔍 I identified a {search_part}, but no exact match for {search_model} in inventory."

            log_partfindr_session(
                {
                    "whatsappNumber": sender_id,
                    "imageUrl": media_url,
                    "identifiedPart": {
                        "name": search_part.title(),
                        "category": "unknown",
                        "confidence": 0.8,
                    },
                    "searchQuery": f"{search_part} {search_model}".strip(),
                    "resultsReturned": len(backend_results),
                    "wasSuccessful": bool(backend_results),
                }
            )

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

            backend_results = search_partfindr_backend(f"{search_part} {search_model}".strip())

            if backend_results:
                reply = format_backend_result(backend_results)
            else:
                best_match, highest_score = local_best_match(search_part, search_model)
                if best_match and highest_score > 60:
                    reply = f"✅ Found: {best_match['part_name']} ({best_match['vehicle']})\n💰 Price: N{best_match['price_NGN']:,} \n 📌 Location: {best_match['location']}"
                else:
                    reply = f"Sorry, I couldn't find a {search_part} for {search_model}. We may not have it in stock currently, check back in a few days or Please double check the model name(You may have sent the wrong model or year.)."

            send_whatsapp_message(sender_id, reply)

            log_partfindr_session(
                {
                    "whatsappNumber": sender_id,
                    "imageUrl": "https://partfindr.local/text-query",
                    "identifiedPart": {
                        "name": search_part.title(),
                        "category": "unknown",
                        "confidence": 0.7,
                    },
                    "searchQuery": f"{search_part} {search_model}".strip(),
                    "resultsReturned": len(backend_results),
                    "wasSuccessful": bool(backend_results),
                }
            )

    except Exception:
        traceback.print_exc()

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)