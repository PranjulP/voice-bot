import os
import sys
import json
import tempfile
import time

# Add user site-packages so smallestai is findable
sys.path.insert(0, '/Users/pranjul/Library/Python/3.9/lib/python/site-packages')
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from groq import Groq
# --- API Clients ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Borrower Persona ---
BORROWER_PERSONA = """
You are Priya, a professional telecalling debt collection advisor at a collections company.
You are calling Suresh (the user) about his ₹15,000 personal loan that is 30 days overdue.
Your goal is to get a firm payment commitment — a specific date and amount.

Your style:
- Polite but firm. Never aggressive, never rude.
- Acknowledge the borrower's situation but always bring the conversation back to payment.
- If they say "I'll pay next week", push for a specific date.
- If they give an excuse, empathize briefly then redirect: "I understand, but let's figure out a way to resolve this today."
- Keep responses short 
— 1 to 2 sentences max.
- Each sentence must be under 12 words. Never combine two thoughts into one sentence.
- Never break character. Never mention you are an AI.

Start the conversation by greeting Suresh and stating the purpose of your call.
"""


# --- WebSocket Endpoint ---
# A persistent connection between browser and server.
# Unlike HTTP, we can push multiple messages back without the browser asking.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation_history = []

    try:
        while True:
            # Wait for a message from the browser
            data = await websocket.receive()

            # Browser sends audio as binary bytes
            if "bytes" in data:
                audio_bytes = data["bytes"]
            else:
                continue

            t_start = time.time()

            # --- STEP 1: STT ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en"
                )
            advisor_text = transcription.text
            t_stt = time.time()
            print(f"\n{'='*50}")
            print(f"Advisor: {advisor_text}")
            print(f"[T1] STT done: {t_stt - t_start:.2f}s")

            await websocket.send_text(json.dumps({
                "type": "advisor_text",
                "text": advisor_text
            }))

            # --- STEP 2: LLM Streaming ---
            messages = [{"role": "system", "content": BORROWER_PERSONA}]
            messages += conversation_history
            messages.append({"role": "user", "content": advisor_text})

            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.8,
                stream=True
            )

            # --- STEP 3: Sentence Detection + TTS per sentence ---
            sentence_buffer = ""
            full_response = ""
            first_token_logged = False
            first_sentence_logged = False

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                sentence_buffer += token
                full_response += token

                # Log time to first token
                if token and not first_token_logged:
                    t_first_token = time.time()
                    print(f"[T2] First Groq token: {t_first_token - t_stt:.2f}s")
                    first_token_logged = True

                stripped = sentence_buffer.strip()
                if stripped and stripped[-1] in [".", "?", "!"] and len(stripped) > 8:
                    t_sentence = time.time()
                    if not first_sentence_logged:
                        print(f"[T3] First sentence complete: {t_sentence - t_stt:.2f}s from STT")
                    print(f"Sending to TTS: {stripped}")
                    await send_tts_chunk(websocket, stripped, t_sentence, first_sentence_logged, t_start if not first_sentence_logged else None)
                    first_sentence_logged = True
                    sentence_buffer = ""

            if sentence_buffer.strip():
                await send_tts_chunk(websocket, sentence_buffer.strip(), time.time(), True)

            t_end = time.time()
            print(f"[T5] Full response done: {t_end - t_start:.2f}s total")
            print(f"Borrower: {full_response}")
            print(f"{'='*50}\n")

            # Update conversation history
            conversation_history.append({"role": "user", "content": advisor_text})
            conversation_history.append({"role": "assistant", "content": full_response})

            # Tell browser the response is complete
            await websocket.send_text(json.dumps({
                "type": "borrower_text",
                "text": full_response
            }))
            await websocket.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        print("Browser disconnected")


async def send_tts_chunk(websocket: WebSocket, text: str, t_before: float, already_logged: bool, t_start: float = None):
    """Convert a sentence to audio and push it to the browser immediately."""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
        response_format="mp3"
    )
    audio = response.content
    t_tts_done = time.time()
    if not already_logged:
        print(f"[T4] First TTS chunk ready: {t_tts_done - t_before:.2f}s for TTS")
        if t_start:
            print(f"[TTFA] Time to first audio (user stop → audio out): {t_tts_done - t_start:.2f}s ⬅ key metric")
    await websocket.send_bytes(audio)


@app.get("/")
async def root():
    return FileResponse("index.html")
