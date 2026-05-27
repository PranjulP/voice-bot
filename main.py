import os
import json
import uuid
import tempfile
import time
import sqlite3
import httpx
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from groq import Groq

# --- API Clients ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")

# --- Paths ---
DATA_DIR = Path("/data")
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "calls.db"
DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PROMPT = """You are Priya, a professional telecalling debt collection advisor at a collections company.
You are calling Suresh (the user) about his ₹15,000 personal loan that is 30 days overdue.
Your goal is to get a firm payment commitment — a specific date and amount.

Your style:
- Polite but firm. Never aggressive, never rude.
- Acknowledge the borrower's situation but always bring the conversation back to payment.
- If they say "I'll pay next week", push for a specific date.
- If they give an excuse, empathize briefly then redirect: "I understand, but let's figure out a way to resolve this today."
- Keep responses short — 1 to 2 sentences max.
- Each sentence must be under 12 words. Never combine two thoughts into one sentence.
- Never break character. Never mention you are an AI.

Start the conversation by greeting Suresh and stating the purpose of your call."""


# --- DB ---
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS config (
            id INTEGER PRIMARY KEY DEFAULT 1,
            stt_provider TEXT DEFAULT 'openai',
            stt_model TEXT DEFAULT 'whisper-1',
            stt_language TEXT DEFAULT 'en',
            stt_temperature REAL DEFAULT 0.0,
            llm_provider TEXT DEFAULT 'openai',
            llm_model TEXT DEFAULT 'gpt-4o-mini',
            llm_temperature REAL DEFAULT 0.8,
            llm_max_tokens INTEGER DEFAULT 150,
            llm_top_p REAL DEFAULT 1.0,
            llm_freq_penalty REAL DEFAULT 0.0,
            llm_presence_penalty REAL DEFAULT 0.0,
            tts_voice TEXT DEFAULT 'aura-asteria-en',
            system_prompt TEXT DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            turn_number INTEGER,
            user_text TEXT,
            bot_text TEXT,
            ttfa_ms INTEGER,
            stt_ms INTEGER,
            llm_ms INTEGER,
            tts_ms INTEGER,
            user_audio_path TEXT,
            bot_audio_path TEXT
        )
    """)
    c.execute("INSERT OR IGNORE INTO config (id) VALUES (1)")
    c.execute("UPDATE config SET system_prompt = ? WHERE id = 1 AND (system_prompt IS NULL OR system_prompt = '')", (DEFAULT_PROMPT,))
    conn.commit()
    conn.close()

def load_config():
    conn = get_db()
    row = conn.execute("SELECT * FROM config WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else {}


# --- Auth ---
def check_auth(request: Request):
    auth = request.headers.get("X-Admin-Password", "")
    if auth != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- STT ---
async def run_stt(audio_bytes: bytes, config: dict) -> str:
    provider = config.get("stt_provider", "openai")
    model = config.get("stt_model", "whisper-1")
    language = config.get("stt_language", "en") or "en"
    temperature = float(config.get("stt_temperature", 0.0))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    if provider == "openai":
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                temperature=temperature
            )
        return result.text

    elif provider == "deepgram":
        params = f"model={model}&language={language}&punctuate=true&smart_format=true"
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        response = httpx.post(
            f"https://api.deepgram.com/v1/listen?{params}",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/webm"
            },
            content=audio_data,
            timeout=15.0
        )
        result = response.json()
        return result["results"]["channels"][0]["alternatives"][0]["transcript"]

    elif provider == "groq":
        with open(tmp_path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model=model,
                file=("audio.webm", f, "audio/webm"),
                language=language,
                temperature=temperature
            )
        return result.text

    return ""


# --- LLM ---
def run_llm_stream(messages: list, config: dict):
    provider = config.get("llm_provider", "openai")
    model = config.get("llm_model", "gpt-4o-mini")
    temperature = float(config.get("llm_temperature", 0.8))
    max_tokens = int(config.get("llm_max_tokens", 150))
    top_p = float(config.get("llm_top_p", 1.0))

    if provider == "groq":
        return groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )
    else:  # openai default
        return openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=float(config.get("llm_freq_penalty", 0.0)),
            presence_penalty=float(config.get("llm_presence_penalty", 0.0)),
            stream=True
        )


# --- TTS ---
async def run_tts(text: str, config: dict) -> bytes:
    voice = config.get("tts_voice", "aura-asteria-en")
    response = httpx.post(
        f"https://api.deepgram.com/v1/speak?model={voice}&encoding=mp3",
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"text": text},
        timeout=10.0
    )
    if response.status_code != 200:
        print(f"TTS error {response.status_code}: {response.text[:300]}")
    return response.content


# --- WebSocket ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    config = load_config()
    conversation_history = []
    session_id = str(uuid.uuid4())
    turn_number = 0

    session_audio_dir = AUDIO_DIR / session_id
    session_audio_dir.mkdir(exist_ok=True)

    try:
        while True:
            data = await websocket.receive()
            if "bytes" not in data:
                continue

            audio_bytes = data["bytes"]
            turn_number += 1
            t_start = time.time()

            # Save user audio
            user_audio_path = str(session_audio_dir / f"user_{turn_number}.webm")
            with open(user_audio_path, "wb") as f:
                f.write(audio_bytes)

            # --- STT ---
            advisor_text = await run_stt(audio_bytes, config)
            t_stt = time.time()
            stt_ms = int((t_stt - t_start) * 1000)
            print(f"\n{'='*50}")
            print(f"[{session_id[:8]}] Turn {turn_number} | STT: {config['stt_provider']}/{config['stt_model']}")
            print(f"Advisor: {advisor_text}")
            print(f"[T1] STT: {stt_ms}ms")

            await websocket.send_text(json.dumps({"type": "advisor_text", "text": advisor_text}))

            # --- LLM ---
            system_prompt = config.get("system_prompt") or DEFAULT_PROMPT
            messages = [{"role": "system", "content": system_prompt}]
            messages += conversation_history
            messages.append({"role": "user", "content": advisor_text})

            stream = run_llm_stream(messages, config)

            sentence_buffer = ""
            full_response = ""
            first_token_logged = False
            first_sentence_logged = False
            first_audio_sent = False
            bot_audio_chunks = []
            ttfa_ms = 0
            tts_ms = 0
            t_llm_first_sentence = None

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                sentence_buffer += token
                full_response += token

                if token and not first_token_logged:
                    print(f"[T2] First token: {int((time.time() - t_stt)*1000)}ms ({config['llm_provider']})")
                    first_token_logged = True

                stripped = sentence_buffer.strip()
                if stripped and stripped[-1] in [".", "?", "!"] and len(stripped) > 8:
                    t_sentence = time.time()
                    if not first_sentence_logged:
                        t_llm_first_sentence = t_sentence
                        print(f"[T3] First sentence: {int((t_sentence - t_stt)*1000)}ms from STT")
                    first_sentence_logged = True

                    t_tts_start = time.time()
                    audio = await run_tts(stripped, config)
                    t_tts_done = time.time()
                    bot_audio_chunks.append(audio)

                    if not first_audio_sent:
                        ttfa_ms = int((t_tts_done - t_start) * 1000)
                        tts_ms = int((t_tts_done - t_tts_start) * 1000)
                        print(f"[T4] TTS: {tts_ms}ms")
                        print(f"[TTFA] {ttfa_ms}ms ⬅ key metric")
                        first_audio_sent = True

                    await websocket.send_bytes(audio)
                    sentence_buffer = ""

            # Flush remaining buffer
            if sentence_buffer.strip():
                t_tts_start = time.time()
                audio = await run_tts(sentence_buffer.strip(), config)
                bot_audio_chunks.append(audio)
                if not first_audio_sent:
                    ttfa_ms = int((time.time() - t_start) * 1000)
                    tts_ms = int((time.time() - t_tts_start) * 1000)
                    first_audio_sent = True
                await websocket.send_bytes(audio)

            t_end = time.time()
            llm_ms = int(((t_llm_first_sentence or t_end) - t_stt) * 1000)
            print(f"[T5] Total: {int((t_end - t_start)*1000)}ms")
            print(f"Borrower: {full_response}")

            # Save bot audio
            bot_audio_path = str(session_audio_dir / f"bot_{turn_number}.mp3")
            with open(bot_audio_path, "wb") as f:
                for chunk_data in bot_audio_chunks:
                    f.write(chunk_data)

            # Log to DB
            conn = get_db()
            conn.execute("""
                INSERT INTO calls (session_id, turn_number, user_text, bot_text,
                    ttfa_ms, stt_ms, llm_ms, tts_ms, user_audio_path, bot_audio_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, turn_number, advisor_text, full_response,
                  ttfa_ms, stt_ms, llm_ms, tts_ms, user_audio_path, bot_audio_path))
            conn.commit()
            conn.close()

            conversation_history.append({"role": "user", "content": advisor_text})
            conversation_history.append({"role": "assistant", "content": full_response})

            await websocket.send_text(json.dumps({"type": "borrower_text", "text": full_response}))
            await websocket.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        print(f"[{session_id[:8]}] Disconnected after {turn_number} turns")


# --- Admin REST Endpoints ---

@app.get("/config")
async def get_config(request: Request):
    check_auth(request)
    return JSONResponse(load_config())

@app.post("/config")
async def post_config(request: Request):
    check_auth(request)
    body = await request.json()
    allowed = [
        "stt_provider", "stt_model", "stt_language", "stt_temperature",
        "llm_provider", "llm_model", "llm_temperature", "llm_max_tokens",
        "llm_top_p", "llm_freq_penalty", "llm_presence_penalty",
        "tts_voice", "system_prompt"
    ]
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields provided")
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    set_clause += ", updated_at = datetime('now')"
    conn = get_db()
    conn.execute(f"UPDATE config SET {set_clause} WHERE id = 1", list(updates.values()))
    conn.commit()
    conn.close()
    return JSONResponse({"status": "ok"})

@app.get("/logs")
async def get_logs(request: Request):
    check_auth(request)
    conn = get_db()
    sessions = conn.execute("""
        SELECT session_id,
               MIN(timestamp) as started_at,
               COUNT(*) as turns,
               ROUND(AVG(ttfa_ms)) as avg_ttfa_ms,
               ROUND(AVG(stt_ms)) as avg_stt_ms,
               ROUND(AVG(llm_ms)) as avg_llm_ms,
               ROUND(AVG(tts_ms)) as avg_tts_ms
        FROM calls
        GROUP BY session_id
        ORDER BY started_at DESC
    """).fetchall()
    result = []
    for s in sessions:
        s_dict = dict(s)
        turns = conn.execute(
            "SELECT * FROM calls WHERE session_id = ? ORDER BY turn_number",
            (s_dict["session_id"],)
        ).fetchall()
        s_dict["turns_detail"] = [dict(t) for t in turns]
        result.append(s_dict)
    conn.close()
    return JSONResponse(result)

@app.get("/audio/{session_id}/{filename}")
async def get_audio(session_id: str, filename: str, request: Request, pw: str = ""):
    # Accept auth via header OR query param (needed for <audio> tags)
    if pw != ADMIN_PASSWORD:
        check_auth(request)
    path = AUDIO_DIR / session_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/webm"
    return FileResponse(str(path), media_type=media_type)

@app.get("/admin")
async def admin_page():
    return FileResponse("admin.html")

@app.get("/")
async def root():
    return FileResponse("index.html")


# --- Init ---
init_db()
