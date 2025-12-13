import os
import json
import requests
import base64
import traceback
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from groq import Groq # Used for the Llama Intelligence Step (Llama 3, free and fast)
from google import genai #Google GenAI SDK
from google.genai import types # To handle multimodal inputs (like images)
import uvicorn

# Load passwords and tokens from .env file
load_dotenv()

app = FastAPI()

# --- 1. INITIALIZATION ---

# Initialize LLM Clients (Reads API keys from .env)
# Groq Client for Llama Intelligence
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Gemini Client for Vision Proxy (Picks up GEMINI_API_KEY from .env automatically)
try:
    gemini_client = genai.Client()
    print("Gemini Client initialized.")
except Exception as e:
    print(f"ERROR: Could not initialize Gemini Client. Check GEMINI_API_KEY: {e}")
    # Initialize with a dummy object to prevent app crash if key is missing
    gemini_client = None 

# Load Inventory
INVENTORY = []
try:
    with open('inventory.json', 'r') as f:
        INVENTORY = json.load(f)
    print(f"Successfully loaded {len(INVENTORY)} items into inventory.")
except Exception as e:
    print(f"ERROR: Could not load inventory.json. Make sure the file exists: {e}")
    # Continue running even if inventory fails to load

# --- 2. WHATSAPP UTILITY FUNCTION ---

def send_whatsapp_message(recipient_id, message_text):
    """Sends a text message back to the user via the WhatsApp Cloud API."""
    
    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN") 
    PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID") 
    
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("ERROR: WHATSAPP_TOKEN or PHONE_NUMBER_ID not set.")
        return {"status": "Error: Missing credentials"}

    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message_text},
    }
    
    response = requests.post(url, headers=headers, json=data)
    print(f"WhatsApp Reply Status: {response.status_code}")
    return response.json()

# --- 3. FASTAPI ENDPOINTS ---

@app.get("/")
async def home():
    """Simple health check endpoint."""
    return {"status": "The Mechanic Bot is Alive and running!"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta Webhook Verification (GET request)."""
    token = os.getenv("VERIFY_TOKEN")
    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and verify_token:
        if mode == "subscribe" and verify_token == token:
            print("Webhook Verified!")
            return int(challenge)
        else:
            print(f"Verification failed. Tokens didn't match: {verify_token} vs {token}")
            raise HTTPException(status_code=403, detail="Verification failed")
    return {"error": "Missing parameters"}

@app.post("/webhook")
async def receive_message(request: Request):
    """Handles incoming messages from WhatsApp (POST request)."""
    data = await request.json()
    
    # 1. Start processing the inbound data structure
    try: 
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
    except (IndexError, KeyError):
        # Ignore malformed or initial test pings
        return {"status": "ok"}
    
    # 2. Check for actual messages (not statuses or other events)
    if "messages" in value:
        try:
            message = value["messages"][0]
            sender_id = message["from"]
            
            if message.get("type") == "image":
                
                # --- START INNER TRY BLOCK (for API/Processing Errors) ---
                try:
                    if not gemini_client:
                        raise Exception("Gemini Client not initialized. Check GEMINI_API_KEY in .env")

                    image_id = message["image"]["id"]
                    print(f"--- STARTING IMAGE PROCESSING for ID: {image_id} from {sender_id} ---")
                    
                    # 1. FETCH IMAGE METADATA (Authenticated by your server)
                    media_url = f"https://graph.facebook.com/v19.0/{image_id}"
                    WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
                    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}",
                               "User-Agent": "Mozilla/5.0"}
                    media_response = requests.get(media_url, headers=headers).json()
                    
                    final_media_url = media_response["url"]
                    print(f"Media URL Fetched: {final_media_url[:50]}...")

                    # 2. DOWNLOAD IMAGE CONTENT (Authenticated by your server)
                    image_data_response = requests.get(final_media_url, headers=headers)
                    image_data_response.raise_for_status() # Raise error if download fails
                    
                    # NOTE: We skip Base64 encoding, as the Gemini SDK can handle raw bytes more easily.
                    image_bytes = image_data_response.content
                    
                    # 4A. VISION STEP (Gemini 2.5 Flash-Lite to describe - Free Tier)
                    VISION_PROMPT = "Describe this car part in detail, including the type, car model, and approximate year if possible. Be concise and return ONLY the descriptive sentence."
                    
                    # Convert image bytes into a Part object for Gemini
                    image_part = types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg' 
                    )

                    # Calling the Gemini API
                    vision_response = gemini_client.models.generate_content(
                        model='gemini-2.5-flash-lite', 
                        contents=[
                            image_part, # The image data part
                            VISION_PROMPT # The text prompt part
                        ],
                    )
                    
                    description = vision_response.text.strip()
                    print(f"Vision Model Description: {description}")


                    # 4B. LLAMA STEP (Intelligence/CRM Decision - using Groq)
                    LLAMA_PROMPT = (
                        f"Analyze the following description: '{description}'. "
                        "Your goal is to extract the single, most relevant car part name and car model for an inventory search. "
                        "**ABSOLUTELY DO NOT use lists, bullet points, or any extra text.** "
                        "Return ONLY the part name and car model in the EXACT format: 'Part Name - Car Model'. "
                        "If the model is unknown, return: 'Part Name - Unknown Model'."
                    )

                    llama_response = groq_client.chat.completions.create( # <--- Using Groq Client
                        model="llama-3.1-8b-instant", # <--- Groq's Llama 3 Model
                        messages=[
                            {"role": "system", "content": "You are a precise part name extractor."},
                            {"role": "user", "content": LLAMA_PROMPT}
                        ],
                    )
                    
                    # Extracting the content, take the first line, and clean it up
                    part_name_raw = llama_response.choices[0].message.content
                    part_name = part_name_raw.strip().split('\n')[0]
                    print(f"Llama 3 Extracted for Search (Cleaned): {part_name}")
                    
                    # 5. REAL Inventory Search Logic
                    found_part = None
                    llama_output = part_name.strip().lower() # e.g., "alternator - honda accord"

                    # Attempt to split the Llama output into part and model
                    try:
                        search_part, search_model = llama_output.split(' - ', 1)
                    except ValueError:
                        # Fallback if Llama failed to use the requested format
                        search_part = llama_output 
                        search_model = ""

                    print(f"Searching for Part: '{search_part}' and Model: '{search_model}'")

                    for item in INVENTORY:
                        item_part = item['part_name'].lower()
                        item_vehicle = item['vehicle'].lower()
                        
                        # Check if the extracted part is a substring of the inventory part name
                        part_match = search_part in item_part
                        
                        # Check if the extracted model is a substring of the inventory vehicle name
                        # The search model must be present and not just an empty string for a match
                        model_match = search_model and (search_model in item_vehicle or search_model in item_part)
                        
                        # We consider it a match if the part is found AND the model is either found or was "unknown"
                        if part_match and (not search_model or model_match):
                            found_part = item
                            break

                    # 6. Construct and Send Reply
                    if found_part:
                        reply_text = (
                            f"✅ IDENTIFIED: {found_part['part_name']} ({found_part['year']}).\n"
                            f"Price: N{found_part['price_NGN']:,.0f}.\n"
                            f"Stock: {found_part['stock_qty']} available. Located at {found_part['location']}.\n"
                            "Do you want me to reserve this for you?"
                        )
                    else:
                        # --- MODIFIED LOGIC IF THE PART WAS IDENTIFIED BUT MODEL UNKNOWN ---
                        if "unknown model" in llama_output:
                            # If model is unknown, specifically ask the user for the vehicle model/year
                            reply_text = (
                                f"🔍 Identified the part as: {search_part.title()}.\n"
                                "However, the vehicle model could not be determined from the image.\n\n"
                                "To find the correct part, **please reply with the exact make and year of your vehicle** (e.g., 'Lexus RX350 2015')."
                            )
                        else:
                            # If the model was known but still didn't match inventory
                            reply_text = (
                                f"🔍 Identified: {part_name}. I can't find an exact match in our inventory right now.\n"
                                "Please try another angle or text the part number for a manual search."
                            )
                        # --- MODIFIED LOGIC ENDS HERE ---
                        
                    send_whatsapp_message(sender_id, reply_text)
                    
                # --- END INNER TRY BLOCK, START INNER EXCEPT ---
                except Exception as e:
                    print(f"\nFATAL RUNTIME ERROR during Image Processing: {e}")
                    traceback.print_exc()
                    send_whatsapp_message(sender_id, f"Critical Processing Error: {e}. Check your keys and network.")
                    
                return {"status": "Image processed"}
            
            # --- TEXT MESSAGE PROCESSING ---
            elif message.get("type") == "text":
                text_body = message["text"]["body"]
                # For text, we can directly send the query to Llama 3
                
                LLAMA_TEXT_PROMPT = (
                    f"A user is looking for a car part with the following query: '{text_body}'. "
                    "Analyze the query and extract the single, most relevant part name and model for an inventory search. "
                    "Return ONLY the part name and model in the format: 'Part Name - Car Model'. Example: 'Alternator - Honda Accord 2006'."
                )

                try:
                    llama_text_response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", # Using the stable 'instant' model
                        messages=[
                            {"role": "system", "content": "You are a precise part name extractor."},
                            {"role": "user", "content": LLAMA_TEXT_PROMPT}
                        ],
                    )
                    
                    # Extract the content, take the first line, and clean it up (ROBUST EXTRACTION)
                    text_part_name_raw = llama_text_response.choices[0].message.content
                    text_part_name = text_part_name_raw.strip().split('\n')[0]
                    
                    # 5. Inventory Search for Text Query (Using same logic as image)
                    found_part = None
                    llama_output = text_part_name.strip().lower()

                    # Attempt to split the Llama output into part and model
                    try:
                        search_part, search_model = llama_output.split(' - ', 1)
                    except ValueError:
                        # Fallback if Llama failed to use the requested format
                        search_part = llama_output 
                        search_model = ""
                        

                    print(f"Text searching for Part: '{search_part}' and Model: '{search_model}'")

                    for item in INVENTORY:
                        item_part = item['part_name'].lower()
                        item_vehicle = item['vehicle'].lower()
                        
                        part_match = search_part in item_part
                        model_match = search_model and (search_model in item_vehicle or search_model in item_part)
                        
                        # Same logic: part must be found, and model must match if it was specified
                        if part_match and (not search_model or model_match):
                            found_part = item
                            break
                    
                    if found_part:
                        reply_text = (
                            f"✅ TEXT SEARCH: {found_part['part_name']} ({found_part['year']}) found.\n"
                            f"Price: N{found_part['price_NGN']:,.0f}.\n"
                            f"Stock: {found_part['stock_qty']} available."
                        )
                    else:
                        reply_text = (
                            f"🔍 TEXT SEARCH: Identified {text_part_name}. No exact match in inventory.\n"
                            "Please try sending a photo for better identification."
                        )
                    
                    send_whatsapp_message(sender_id, reply_text)
                    
                except Exception as e:
                    print(f"ERROR processing text message with Llama: {e}")
                    traceback.print_exc()
                    send_whatsapp_message(sender_id, "Sorry, I ran into an error processing your text query.")

                return {"status": "Text message received"}
                
        # --- END OUTER TRY BLOCK for message parsing, START OUTER EXCEPT ---
        except Exception as e:
            print(f"ERROR: Failed to parse WhatsApp message payload: {e}")
            traceback.print_exc()
            return {"status": "Message parse failed"}

    # 3. Handle Statuses and non-message events
    elif "statuses" in value:
        return {"status": "Status received, ignoring"}
    
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)