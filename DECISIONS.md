# Voice Bot — Decision Log

Last updated: 2026-05-27

---

## v2 Design — Config + Logging System

### Architecture
- Same Railway deployment, no new services
- SQLite DB on Railway Volume at `/data/calls.db`
- Audio files saved to `/data/audio/{session_id}/`
- Config changes apply on **next call** (not mid-call)

### Pages
- `/` → public call UI, no auth
- `/admin` → password protected (env var: `ADMIN_PASSWORD`)

---

## Provider Options (exposed in Admin UI)

### STT
| Provider | Models | Extra params |
|---|---|---|
| OpenAI Whisper | `whisper-1` | language, temperature |
| Deepgram Nova-2 | `nova-2`, `nova-2-phonecall`, `nova-2-general` | language |
| Groq Whisper | `whisper-large-v3`, `whisper-large-v3-turbo`, `distil-whisper-large-v3-en` | language, temperature |

- Integration: REST API (not WebSocket streaming) for simplicity. ~100ms overhead acceptable for now.
- WebSocket streaming deferred to when moving to telephony infra.

### LLM
| Provider | Models |
|---|---|
| OpenAI | `gpt-4o-mini`, `gpt-4o` |
| Groq | `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768` |

- Configurable: temperature, max_tokens, top_p, frequency_penalty (OpenAI only), presence_penalty (OpenAI only)

### TTS
| Provider | Voices |
|---|---|
| Deepgram | `aura-asteria-en`, `aura-luna-en`, `aura-stella-en`, `aura-hera-en`, `aura-orion-en`, `aura-zeus-en` |

- Encoding fixed to `mp3` (browser compatibility)
- Other providers (Sarvam AI for Indian accent) deferred

---

## Call Recording
- **Decision**: Save both user audio and bot audio
- User audio: `.webm` per turn
- Bot audio: `.mp3` per turn (concatenated TTS chunks)
- Path: `/data/audio/{session_id}/user_{turn}.webm` and `bot_{turn}.mp3`

---

## Database Schema

### `config` table (single row, updated in place)
```
stt_provider, stt_model, stt_language, stt_temperature,
llm_provider, llm_model, llm_temperature, llm_max_tokens, llm_top_p, llm_freq_penalty, llm_presence_penalty,
tts_voice,
system_prompt,
updated_at
```

### `calls` table (one row per turn)
```
id, session_id, timestamp, turn_number,
user_text, bot_text,
ttfa_ms, stt_ms, llm_ms, tts_ms,
user_audio_path, bot_audio_path
```

---

## Admin UI — 3 Tabs

1. **Config** — STT / LLM / TTS dropdowns + params + system prompt textarea + Save
2. **Logs** — Table by session. Expand to see transcript + audio player per turn
3. **Tab 3** — Placeholder (future: persona builder / test runner)

---

## Prompt / Persona
- Basic textarea in admin UI for now
- Full persona builder deferred to later

---

## Railway Setup Required (one-time, done by Pranjul)
1. Service → Volumes → Add volume, mount at `/data`
2. Add env var: `ADMIN_PASSWORD=yourpassword`

---

## Build Order
1. Railway Volume setup (Pranjul)
2. SQLite schema + DB init in `main.py`
3. Dynamic STT/LLM/TTS routing based on config
4. Per-call logging + audio saving
5. REST endpoints (`/config`, `/logs`, `/audio`)
6. `admin.html` UI

---

## Open / Deferred
- Indian accent TTS (Sarvam AI) — deferred
- Deepgram WebSocket streaming STT — deferred
- Mid-call config changes — not needed
- Persona builder beyond basic textarea — deferred
- STT latency fix (currently 1.6s, target <2s) — deferred to after v2
