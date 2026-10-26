import asyncio
import gc
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rednote-mm-bot")


def setup_ffmpeg() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = "/tmp/ffbin"
        os.makedirs(bin_dir, exist_ok=True)
        target = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(target):
            try:
                os.symlink(exe, target)
            except Exception:
                shutil.copy2(exe, target)
                os.chmod(target, 0o755)
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        return target
    except Exception:
        return "ffmpeg"


FFMPEG_BIN = setup_ffmpeg()
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else "/tmp"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB").strip()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://rednote-bot-gzd0.onrender.com").strip()

ADMIN_ID = int(os.environ.get("ADMIN_ID", "8430004524").strip() or "8430004524")
ALLOWED_USER_IDS = {8855067536}

SAMPLE_RATE = 24000
MAX_TG_BYTES = 45 * 1024 * 1024
CHUNK_DURATION = 300
PAUSE_AFTER_SCENE = 0.20

PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
os.makedirs(PUBLISH_CACHE_DIR, exist_ok=True)

CANCELLED_JOBS: set = set()
video_render_semaphore = asyncio.Semaphore(1)

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
PREFERRED_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"]
SPINNING_EMOJIS = ["⏳", "⌛", "⏳", "⌛"]

DEEP_VOICE_FILTER = (
    "highpass=f=55,"
    "asetrate=22080,aresample=24000,atempo=1.0870,"
    "equalizer=f=110:t=q:w=1.0:g=4.5,"
    "equalizer=f=230:t=q:w=1.1:g=2.5,"
    "equalizer=f=6500:t=q:w=1.0:g=-1.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def get_anim_emoji(frame: int) -> str:
    return SPINNING_EMOJIS[frame % len(SPINNING_EMOJIS)]


def fmt_progress_bar(pct: int) -> str:
    pct = max(0, min(100, pct))
    filled = int(round(pct / 10))
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_cmd(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        out, err = proc.communicate()
        return subprocess.CompletedProcess(cmd, -9, out, err)


class ProgressTracker:
    def __init__(self, chat_id: int, message_id: int, job_id: str = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.job_id = job_id
        self.last_ts = 0.0
        self.last_text = ""
        self.anim_frame = 0

    async def update(self, text: str, force: bool = False):
        if not self.message_id or text == self.last_text:
            return
        now = time.monotonic()
        if not force and (now - self.last_ts) < 3.0:
            return
        self.last_ts = now
        self.last_text = text
        self.anim_frame += 1
        animated_text = f"{get_anim_emoji(self.anim_frame)} {text}"
        try:
            await tg_api("editMessageText", {
                "chat_id": self.chat_id,
                "message_id": self.message_id,
                "text": animated_text,
            })
        except Exception:
            pass


async def tg_api(method: str, data: dict = None, timeout: float = 90.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                resp = await client.post(url, json=data or {})
                return resp.json()
        except Exception:
            await asyncio.sleep(1.5)
    return {"ok": False}


async def send_text(chat_id: int, text: str, reply_to: int = None):
    res = await tg_api("sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "reply_to_message_id": reply_to,
    })
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str):
    if not message_id:
        return
    try:
        await tg_api("editMessageText", {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        })
    except Exception:
        pass


async def tg_upload_media(method: str, data: dict, field_name: str, file_name: str, file_path: str, mime_type: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    payload_bytes = await asyncio.to_thread(lambda: open(file_path, "rb").read())
    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    resp = await client.post(
                        url,
                        data=data,
                        files={field_name: (file_name, payload_bytes, mime_type)},
                    )
                    res = resp.json()
                    if res.get("ok"):
                        return res
            except Exception:
                pass
            await asyncio.sleep(2.5)
        return {"ok": False}
    finally:
        del payload_bytes
        gc.collect()


def probe_video_info(path: str) -> tuple[float, int, int]:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        dur = float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    w, h = 0, 0
    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d{2,5})x(\d{2,5})", out)
    if vm:
        w, h = int(vm.group(2)), int(vm.group(3))
    return dur, w, h


def compute_output_dims(src_w: int, src_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1280, 720
    long_edge = max(src_w, src_h)
    scale = 1280.0 / float(long_edge)
    out_w = max(2, int(round(src_w * scale / 2.0)) * 2)
    out_h = max(2, int(round(src_h * scale / 2.0)) * 2)
    return out_w, out_h


async def get_source_duration(url: str) -> float:
    try:
        cmd = ["yt-dlp", "-j", "--no-warnings", url]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if out:
            data = json.loads(out.decode().strip().split("\n")[0])
            dur = data.get("duration")
            if dur:
                return float(dur)
    except Exception as e:
        logger.warning(f"get_source_duration error: {e}")
    return 0


async def download_full(url: str, out_path: str, tracker=None, total_dur: float = 0) -> float:
    cmd = [
        "yt-dlp", "--no-warnings", "--no-part",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", out_path, url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )

    last_report = time.monotonic()
    while True:
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
            break
        except asyncio.TimeoutError:
            if tracker and time.monotonic() - last_report >= 4.0:
                last_report = time.monotonic()
                if os.path.exists(out_path):
                    mb = os.path.getsize(out_path) / (1024 * 1024)
                    pct = min(95, int(mb / 10)) if total_dur <= 0 else 10
                    if total_dur > 0:
                        pct = min(95, int((mb / max(1, total_dur)) * 100 / 4))
                    await tracker.update(
                        f"📥 မူရင်းဗီဒီယို ဒေါင်းနေသည်...\n"
                        f"📦 {mb:.0f} MB\n"
                        f"📊 {fmt_progress_bar(pct)}",
                        force=True,
                    )

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
        return 0
    actual_dur, _, _ = probe_video_info(out_path)
    return actual_dur


async def groq_whisper_transcribe(audio_path: str) -> list:
    if not GROQ_API_KEY or not os.path.exists(audio_path):
        return []
    if os.path.getsize(audio_path) > 24 * 1024 * 1024:
        return []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as af:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
                    files={"file": ("speech.mp3", af, "audio/mpeg")}
                )
            if resp.status_code == 200:
                return [
                    {"start": round(float(s.get("start", 0)), 2), "end": round(float(s.get("end", 0)), 2), "text": s.get("text", "").strip()}
                    for s in resp.json().get("segments", []) if (s.get("text") or "").strip()
                ]
    except Exception as e:
        logger.warning(f"Groq Whisper error: {e}")
    return []


def make_keep_ranges(duration: float, cuts: list, max_remove_ratio: float = 0.08):
    candidates = []
    for item in cuts:
        try:
            start = max(0.0, min(duration, float(item["start"])))
            end = max(0.0, min(duration, float(item["end"])))
            if end > start:
                candidates.append((start, end))
        except (KeyError, TypeError, ValueError):
            continue

    candidates.sort()
    merged = []
    for start, end in candidates:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    removal_limit = duration * max_remove_ratio
    accepted_cuts = []
    removed = 0.0
    for start, end in merged:
        remaining = removal_limit - removed
        if remaining <= 0:
            break
        cut_len = min(end - start, remaining)
        if cut_len > 0:
            accepted_cuts.append((start, start + cut_len))
            removed += cut_len

    keep_ranges = []
    cursor = 0.0
    for start, end in accepted_cuts:
        if start > cursor:
            keep_ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep_ranges.append((cursor, duration))
    return keep_ranges


def split_pcm_by_weights(master: np.ndarray, scenes: list) -> list[np.ndarray]:
    if len(scenes) == 0:
        return []
    if len(scenes) == 1 or len(master) < SAMPLE_RATE:
        return [master]

    weights = np.array([max(5, len(s.get("burmese", ""))) for s in scenes], dtype=np.float64)
    cum_ratios = np.cumsum(weights)[:-1] / np.sum(weights)

    hop = int(0.02 * SAMPLE_RATE)
    n_frames = max(1, (len(master) - hop) // hop)
    rms = np.array([
        float(np.sqrt(np.mean(master[i * hop:i * hop + hop] ** 2)))
        for i in range(n_frames)
    ])

    cuts = [0]
    for r in cum_ratios:
        expected = int(r * n_frames)
        win = max(25, int(0.15 * n_frames))
        lo = max(cuts[-1] // hop + 10, expected - win)
        hi = min(n_frames - 10, expected + win)
        if hi <= lo:
            best = max(cuts[-1] // hop + 1, expected)
        else:
            dists = np.abs(np.arange(lo, hi) - expected) / max(1.0, float(win))
            score = rms[lo:hi] + 0.015 * dists
            best = lo + int(np.argmin(score))
        cuts.append(min(len(master), best * hop))
    cuts.append(len(master))

    clips = []
    for i in range(len(scenes)):
        seg = master[cuts[i]:cuts[i + 1]].copy()
        if len(seg) < 200:
            seg = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        clips.append(seg)
    return clips


async def _tts_elevenlabs_batch(batch: list, batch_idx: int, total_batches: int, work_dir: str):
    if not ELEVENLABS_API_KEY:
        return None
    joined = "\n\n".join(s["burmese"] for s in batch)
    if len(joined) > 4500:
        return None

    tmp_raw = os.path.join(work_dir, f"el_{batch_idx}_raw.mp3")
    tmp_pcm = os.path.join(work_dir, f"el_{batch_idx}.pcm")

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": joined,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {resp.text[:200]}")
            with open(tmp_raw, "wb") as f:
                f.write(resp.content)

        result = run_cmd([
            FFMPEG_BIN, "-y",
            "-i", tmp_raw,
            "-af", DEEP_VOICE_FILTER,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            tmp_pcm,
        ], 180)

        if result.returncode != 0 or not os.path.exists(tmp_pcm):
            raise RuntimeError("ffmpeg convert failed")

        master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        clips = split_pcm_by_weights(master, batch)
        logger.info(f"ElevenLabs batch {batch_idx + 1}/{total_batches} OK")
        return clips
    except Exception as e:
        logger.warning(f"ElevenLabs batch {batch_idx + 1} failed: {str(e)[:200]}")
        return None
    finally:
        for p in (tmp_raw, tmp_pcm):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


async def _deepseek_script(prompt: str) -> str | None:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a professional Myanmar storyteller. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 8192,
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else None
    except Exception as e:
        logger.warning(f"DeepSeek script failed: {str(e)[:200]}")
        return None


async def _tts_one_batch(batch: list, batch_idx: int, total_batches: int, work_dir: str):
    client = genai.Client(api_key=GEMINI_API_KEY)
    joined = "\n\n".join(s["burmese"] for s in batch)

    for model in PREFERRED_TTS_MODELS:
        for attempt in range(2):
            tmp_raw = os.path.join(work_dir, f"tts_{batch_idx}_raw.bin")
            tmp_pcm = os.path.join(work_dir, f"tts_{batch_idx}.pcm")
            try:
                cfg = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                        )
                    ),
                )
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=f"""
အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် သဘာဝကျသော စကားပြောဟန်ဖြင့် ဖတ်ပါ။
အသံသည် နက်ရှိုင်းသော (deep, low), ရင်အုပ်သံပါသော, ပဲ့တင်သံမပါသော အသံဖြစ်ရမည်။
ဇာတ်လမ်းကို ကိုယ်တိုင်ကြုံတွေ့ခဲ့သော အဓိကဇာတ်ကောင်က ပြန်ပြောပြနေသလို အသံသွင်းပါ။
အလွန်မြန်မြန်၊ တစ်သံတည်း မဖတ်ပါနှင့်။
စာသားတွင် မပါသော စကားလုံးများကို မထည့်ပါနှင့်။

{joined}
""",
                    config=cfg,
                )
                cands = getattr(resp, "candidates", None) or []
                if not cands or not getattr(cands[0], "content", None):
                    raise RuntimeError("no candidates")
                parts = cands[0].content.parts
                if not parts or not getattr(parts[0], "inline_data", None):
                    raise RuntimeError("no inline_data")
                data = parts[0].inline_data.data
                if not data or len(data) < 1000:
                    raise RuntimeError(f"too small ({len(data) if data else 0} bytes)")

                with open(tmp_raw, "wb") as f:
                    f.write(data)

                result = run_cmd([
                    FFMPEG_BIN, "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", tmp_raw,
                    "-af", DEEP_VOICE_FILTER,
                    "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    tmp_pcm,
                ], 180)

                if result.returncode != 0 or not os.path.exists(tmp_pcm):
                    raise RuntimeError("ffmpeg convert failed")
                if os.path.getsize(tmp_pcm) < 400:
                    raise RuntimeError("pcm too small")

                master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                clips = split_pcm_by_weights(master, batch)
                logger.info(f"TTS batch {batch_idx + 1}/{total_batches} OK ({model})")
                return clips
            except Exception as e:
                err = str(e)[:200]
                logger.warning(f"TTS batch {batch_idx + 1} attempt {attempt + 1} ({model}): {err}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    await asyncio.sleep(8)
                else:
                    await asyncio.sleep(2)
            finally:
                for p in (tmp_raw, tmp_pcm):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
    return None


async def synthesize_tts(scenes: list, work_dir: str) -> list[np.ndarray]:
    BATCH_SIZE = 6
    batches = [scenes[i:i + BATCH_SIZE] for i in range(0, len(scenes), BATCH_SIZE)]
    logger.info(f"TTS: {len(scenes)} scenes in {len(batches)} batches")
    all_clips = []

    for b_idx, batch in enumerate(batches):
        clips = None
        if ELEVENLABS_API_KEY:
            clips = await _tts_elevenlabs_batch(batch, b_idx, len(batches), work_dir)
        if clips is None:
            clips = await _tts_one_batch(batch, b_idx, len(batches), work_dir)
        if clips is None:
            logger.error(f"All TTS failed for batch {b_idx + 1}/{len(batches)} — silence fallback")
            clips = [np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32) for _ in batch]
        all_clips.extend(clips)
        await asyncio.sleep(1.5)

    return all_clips


async def extract_story_summary(scenes: list, client) -> dict:
    if not scenes:
        return {"summary": "", "characters": [], "last_scene": "", "protagonist": ""}
    all_text = "\n".join(s["burmese"] for s in scenes)
    prompt = f"""
အောက်ပါ မြန်မာဇာတ်လမ်းအပိုင်းကို ဖတ်ပြီး နောက်အပိုင်းအတွက် အကျဉ်းချုပ် ထုတ်ပေးပါ။

{all_text}

Return JSON:
{{"characters": ["နာမည်1"], "protagonist": "အဓိကဇာတ်ကောင်", "summary": "အဓိကအကြောင်းအရာ ၂-၃ ကြောင်း", "last_scene": "နောက်ဆုံးအခြေအနေ"}}
    """.strip()
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=PREFERRED_MODELS[0],
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resp.text.strip())
    except Exception as e:
        logger.warning(f"Summary error: {e}")
        return {
            "characters": [],
            "protagonist": "",
            "summary": all_text[-500:],
            "last_scene": scenes[-1]["burmese"],
        }


async def build_chunk_recap(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    story_context: dict,
    chunk_index: int,
    tracker: ProgressTracker,
):
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_fmt = fmt_time(global_start)
    end_fmt = fmt_time(global_end)

    target_chars = int(chunk_dur * 6.5)
    scene_count = max(15, int(chunk_dur / 8))
    chars_per_scene = max(80, target_chars // scene_count)
    protagonist = story_context.get("protagonist", "") if chunk_index > 1 else ""

    if chunk_index == 1:
        context_section = "🔵 ဤသည် ဇာတ်လမ်း၏ အစဖြစ်သည်။ အဓိကဇာတ်ကောင်က ဇာတ်လမ်းကို ပထမလူအနေနဲ့ ပြန်ပြောပြပါ။"
    else:
        chars_str = ", ".join(story_context.get("characters", [])[:10])
        protag_line = f"- အဓိကဇာတ်ကောင် (မပြောင်းရ): {protagonist}\n" if protagonist else ""
        context_section = f"""
🔵 ယခင်ဖြစ်ပျက်ခဲ့သည့် ဇာတ်လမ်းအကျဉ်း —
{protag_line}- ဇာတ်ကောင်များ: {chars_str}
- အဓိကဖြစ်ရပ်များ: {story_context.get('summary', '')}
- နောက်ဆုံးအခြေအနေ: {story_context.get('last_scene', '')}

🔵 ရှေ့အပိုင်းနဲ့ ရှုထောင့်နဲ့ အဓိကဇာတ်ကောင် လုံးဝ မပြောင်းပါနှင့်။
"""

    await tracker.update("🎧 Groq Whisper နဲ့ အသံကို စာသားပြောင်းနေပါသည်...", force=True)
    audio_path = os.path.join(os.path.dirname(chunk_path), f"chunk_{chunk_index}_audio.mp3")
    run_cmd([FFMPEG_BIN, "-y", "-i", chunk_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k", audio_path], 300)
    
    whisper_segs = await groq_whisper_transcribe(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    whisper_hint = f"\nWhisper transcript:\n{json.dumps(whisper_segs[:120], ensure_ascii=False)}\n" if whisper_segs else ""

    prompt = f"""
သင်သည် ဗီဒီယိုထဲရှိ ဇာတ်လမ်းကို အဓိကဇာတ်ကောင်ကိုယ်တိုင် ပြန်ပြောပြနေသလို မြန်မာဘာသာဖြင့် ရေးပေးရမည့် ဇာတ်ကြောင်းရေးသူဖြစ်သည်။

ဤဗီဒီယိုသည် မူရင်းဗီဒီယို၏ {start_fmt} မှ {end_fmt} အထိ ဖြစ်သည်။ ဤအပိုင်းကိုသာ ကြည့်ပြီး အောက်ပါညွှန်ကြားချက်များအတိုင်း ဇာတ်ကြောင်းရေးပါ။

{context_section}

{whisper_hint}

## ပြောပြပုံ
- ဇာတ်လမ်းကို အတိုချုပ်မလုပ်ပါနှင့်။ မူရင်းဗီဒီယိုတွင် ဖြစ်ပျက်သည့်အတိုင်း အစဉ်လိုက်၊ အပြည့်အစုံနီးပါး ပြန်ပြောပြပါ။
- အဓိကဇာတ်ကောင်က ကိုယ်တိုင်ကြုံတွေ့ခဲ့သည့်အကြောင်းကို ပြောနေသလို ပထမပုဂ္ဂိုလ်အမြင်ဖြင့် ရေးပါ။ ဥပမာ — "ကျွန်တော်ကတော့...", "ကျွန်မကတော့...", "အဲဒီအချိန်မှာ ကျွန်တော်..."
- ဇာတ်ကောင်၏ အမည် သို့မဟုတ် ကျား/မကို ဗီဒီယိုမှ မသိနိုင်ပါက မခန့်မှန်းပါနှင့်။ မသိသောအမည်ကိုလည်း မတီထွင်ပါနှင့်။
- ဖြစ်ရပ်တစ်ခုမှ နောက်တစ်ခုသို့ "အဲဒီအချိန်မှာတော့...", "ဒါပေမယ့် နောက်ပိုင်းမှာ...", "အဲဒီနောက်မှာ..." ဟူသော စကားဆက်များကို လိုအပ်သလို သဘာဝကျကျ သုံးပါ။ တစ်မျိုးတည်းကို ထပ်ခါထပ်ခါ မသုံးပါနှင့်။
- ဇာတ်ကောင်၏ ရည်ရွယ်ချက်၊ ဆုံးဖြတ်ချက်၊ ခံစားချက်နှင့် ဖြစ်ရပ်များ၏ အကြောင်းရင်း၊ အကျိုးဆက်များကို ရှင်းလင်းစွာ ရေးပါ။
- ဇာတ်ကောင်ကိုယ်တိုင် မမြင်ခဲ့၊ မသိနိုင်ခဲ့သည့်အဖြစ်အပျက်ကို သူ့အမြင်ဖြင့် သိရှိသလို မပြောပါနှင့်။ ဗီဒီယိုမှ အတည်ပြုနိုင်သည့်အချက်များကိုသာ အသုံးပြုပါ။

## ဇာတ်လမ်းကို မဖြတ်တောက်ရန်
- အဓိကဖြစ်ရပ်များအပြင် ဇာတ်လမ်းကို နားလည်ရန်လိုအပ်သော အသေးစိတ်ဖြစ်ရပ်များ၊ ဇာတ်ကောင်များ၏ ဆက်နွယ်မှု၊ စကားပြောချက်များ၏ အဓိပ္ပာယ်၊ အငြင်းပွားမှု၊ လျှို့ဝှက်ချက်၊ အလှည့်အပြောင်းနှင့် အကျိုးဆက်များကို မချန်ပါနှင့်။
- စကားပြောချက်များကို မူရင်းအတိုင်း စာလုံးချင်း ကူးရေးရန်မလိုပါ။ သို့သော် ပြောခဲ့သည့်အကြောင်းအရာ၏ အဓိပ္ပာယ်ကို ထိန်းသိမ်းပါ။
- ဇာတ်လမ်းနှင့် မသက်ဆိုင်သော ရှည်လျားသည့် တိတ်ဆိတ်နေမှု၊ ထပ်နေသော မြင်ကွင်း၊ ထပ်နေသောစကားနှင့် မလိုအပ်သော နားချိန်များကိုသာ ဖယ်ရှားပါ။
- အရေးကြီးသောဖြစ်ရပ်ကို မဖယ်ပါနှင့်။ မူရင်းဗီဒီယိုတွင် မပါသည့်အရာကို မတီထွင်ပါနှင့်။
- ဤအပိုင်းအဆုံးတွင် ဇာတ်လမ်းဆက်နေပါက နောက်ပိုင်းတွင် ဘာဖြစ်မည်ကို ခန့်မှန်းမပြောဘဲ ဤအပိုင်းတွင် မြင်ရသည့်အတိုင်းသာ ရပ်ပါ။

## ကြာချိန်နှင့် ဗီဒီယိုတည်းဖြတ်ရန် အချိန်မှတ်ချက်များ
- မူရင်းအပိုင်း၏ ၉၀ မှ ၉၅ ရာခိုင်နှုန်းခန့်ကို ထိန်းသိမ်းပါ။ အရေးမကြီးသော တိတ်ဆိတ်နေမှု၊ ထပ်နေသောမြင်ကွင်းနှင့် မလိုအပ်သော နားချိန်များကိုသာ ဖြတ်ပါ။
- ဇာတ်ကြောင်းစာသားကို ရည်မှန်းထားသည့် ကြာချိန်နှင့် ကိုက်ညီအောင် လုံလောက်စွာ အသေးစိတ်ရေးပါ။ အလွန်တိုသော အကျဉ်းချုပ် မရေးပါနှင့်။
- အနည်းဆုံး ခန်း {scene_count} ခန်း ရေးပါ။ ခန်းတစ်ခန်းလျှင် အနည်းဆုံး {chars_per_scene} လုံး ပါဝင်ပါ။
- အပိုင်းတစ်ခုစီအတွက် "start" နှင့် "end" တွင် မူရင်းဗီဒီယို၏ ဤအပိုင်းအစမှ စတွက်ထားသော စက္ကန့်ကို ထည့်ပါ။
- အရေးမကြီးသောအပိုင်းကို ဖြတ်သင့်ပါက ၎င်း၏ မူရင်းစတင်ချိန်နှင့် ဆုံးချိန်ကို "cuts" စာရင်းတွင် ထည့်ပြီး ဖြတ်ရသည့်အကြောင်းကို တိုတိုရေးပါ။
- ထိန်းသိမ်းမည့် မြင်ကွင်းများကို မူရင်းအစဉ်အတိုင်း စီပါ။ မြင်ကွင်းအချိန်များ ထပ်မနေစေရ၊ အရေးကြီးသောအပိုင်းများလည်း မကျန်ခဲ့စေရ။

## အဖြေထုတ်ပုံ
အောက်ပါပုံစံအတိုင်း JSON ကိုသာ ပြန်ပေးပါ။ JSON ပြင်ပတွင် ရှင်းလင်းချက် သို့မဟုတ် အခြားစာသား မရေးပါနှင့်။

{{
  "protagonist": "အဓိကဇာတ်ကောင်နာမည် (မသိရင် ကွက်လပ်ထား)",
  "scenes": [
    {{
      "start": 0.0,
      "end": 12.0,
      "burmese": "အဓိကဇာတ်ကောင်၏ အမြင်မှ ရေးထားသော မြန်မာဇာတ်ကြောင်း"
    }}
  ],
  "cuts": [
    {{
      "start": 12.0,
      "end": 15.0,
      "reason": "ထပ်နေသော မြင်ကွင်း"
    }}
  ]
}}
""".strip()

    await tracker.update("🎧 Gemini ဆီ ဗီဒီယိုတင်ပြီး ဇာတ်လမ်းစာသား ရေးနေပါသည်...", force=True)
    video_file = await asyncio.to_thread(client.files.upload, file=chunk_path)

    for _ in range(40):
        state_obj = getattr(video_file, "state", None)
        state = getattr(state_obj, "name", str(state_obj or ""))
        if state in ("ACTIVE", "FAILED"):
            break
        await asyncio.sleep(2)
        video_file = await asyncio.to_thread(client.files.get, name=video_file.name)

    state_obj = getattr(video_file, "state", None)
    state = getattr(state_obj, "name", str(state_obj or ""))
    if state != "ACTIVE":
        raise RuntimeError("Gemini က ဗီဒီယိုကို စစ်ဆေး၍မရပါ။")

    scenes = []
    cuts = []
    new_protagonist = protagonist
    raw_json = None

    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=PREFERRED_MODELS[0],
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        if resp and resp.text:
            raw_json = resp.text.strip()
            logger.info("Script generated by Gemini")
    except Exception as e:
        logger.warning(f"Gemini script failed: {str(e)[:150]}")

    if raw_json is None and DEEPSEEK_API_KEY:
        logger.info("Falling back to DeepSeek")
        deepseek_prompt = prompt + f"\n\nWhisper Transcript:\n{whisper_hint}"
        raw_json = await _deepseek_script(deepseek_prompt)
        if raw_json:
            logger.info("Script generated by DeepSeek")

    if raw_json is not None:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                scenes = parsed.get("scenes", [])
                cuts = parsed.get("cuts", [])
                if not protagonist:
                    new_protagonist = parsed.get("protagonist", "") or protagonist
            elif isinstance(parsed, list):
                scenes = parsed
        except Exception as e:
            logger.error(f"JSON parse error: {e}")

    try:
        await asyncio.to_thread(client.files.delete, name=video_file.name)
    except Exception as e:
        logger.warning(f"Gemini delete error: {e}")

    if not scenes:
        scenes = [{"start": 0, "end": 5, "burmese": "စာသားရယူရာတွင် အမှားအယွင်းရှိပါသည်။"}]

    keep_ranges = make_keep_ranges(chunk_dur, cuts, max_remove_ratio=0.08)
    kept_duration = sum(end - start for start, end in keep_ranges)

    if kept_duration < chunk_dur * 0.90:
        logger.warning(f"Kept {kept_duration:.1f}s < 90%, forcing keep-all")
        keep_ranges = [(0.0, chunk_dur)]
        kept_duration = chunk_dur

    await tracker.update("🎙️ Charon AI / ElevenLabs မှ အသံသွင်းနေပါသည်...", force=True)
    clips = await synthesize_tts(scenes, os.path.dirname(chunk_path))

    out_w, out_h = compute_output_dims(*probe_video_info(chunk_path)[1:])
    
    # 🟢 Copyright Shield (မူပိုင်ခွင့် ကာကွယ်ရေး)
    shield_v = "crop=iw*0.96:ih*0.96,hflip,"
    color = "eq=contrast=1.08:saturation=1.12:gamma=1.02"

    master_pcm_path = os.path.join(os.path.dirname(chunk_path), "master_audio.pcm")
    with open(master_pcm_path, "wb") as pcm_f:
        for clip in clips:
            np.clip(clip * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)
            pause_samples = int(PAUSE_AFTER_SCENE * SAMPLE_RATE)
            if pause_samples > 0:
                np.zeros(pause_samples, dtype=np.int16).tofile(pcm_f)

    master_wav = os.path.join(os.path.dirname(chunk_path), "master_audio.wav")
    run_cmd([
        FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", master_pcm_path, master_wav,
    ], 120)
    if os.path.exists(master_pcm_path):
        os.remove(master_pcm_path)

    seg_files = []
    await tracker.update("🎬 ဗီဒီယို ပြင်ဆင်နေပါသည် (Copyright Shield ဖွင့်ထားသည်)...", force=True)

    for index, (start, end) in enumerate(keep_ranges):
        segment_duration = end - start
        segment_path = os.path.join(os.path.dirname(chunk_path), f"seg_{index:03d}.mp4")
        fc = (
            f"{shield_v}"
            f"fps=25,"
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={out_w}:{out_h},"
            f"{color},format=yuv420p"
        )
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", chunk_path,
            "-t", f"{segment_duration:.3f}",
            "-an",
            "-vf", fc,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            segment_path,
        ], 300)

        if result.returncode != 0 or not os.path.exists(segment_path):
            raise RuntimeError(f"segment {index + 1} failed")
        seg_files.append(segment_path)

    concat_txt = os.path.join(os.path.dirname(chunk_path), "concat.txt")
    with open(concat_txt, "w") as f:
        for tp in seg_files:
            f.write(f"file '{tp}'\n")

    video_only = os.path.join(os.path.dirname(chunk_path), "video_only.mp4")
    await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", video_only,
    ], 300)

    out_chunk = chunk_path.replace(".mp4", "_recap.mp4")
    await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y",
        "-i", video_only,
        "-i", master_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-af", "apad",
        "-t", f"{kept_duration:.3f}",
        "-movflags", "+faststart",
        out_chunk,
    ], 300)

    for f in [chunk_path, video_only, master_wav, concat_txt] + seg_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    new_summary = await extract_story_summary(scenes, client)
    if new_protagonist and not new_summary.get("protagonist"):
        new_summary["protagonist"] = new_protagonist
    return out_chunk, new_summary


async def split_for_telegram(path: str, work_dir: str) -> list[str]:
    if not os.path.exists(path):
        return []
    if os.path.getsize(path) <= MAX_TG_BYTES:
        return [path]

    out_dir = os.path.join(work_dir, "tg_parts")
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "part_%03d.mp4")
    run_cmd([
        FFMPEG_BIN, "-y",
        "-i", path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", "180",
        "-reset_timestamps", "1",
        pattern,
    ], 300)
    parts = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("part_") and f.endswith(".mp4")
    )
    return parts or [path]


async def handle_link_job(chat_id: int, reply_msg_id: int, url: str):
    job_id = f"j{secrets.token_hex(5)}"
    status_id = await send_text(chat_id, "⚡ Recap စတင်နေပါပြီ...", reply_to=reply_msg_id)
    work_dir = tempfile.mkdtemp(prefix="vid_", dir=WORK_ROOT)
    tracker = ProgressTracker(chat_id, status_id, job_id=job_id)

    story_context = {"characters": [], "summary": "", "last_scene": "", "protagonist": ""}

    try:
        await tracker.update("🔍 ဗီဒီယို အချိန်စစ်ဆေးနေပါသည်...", force=True)
        probe_dur = await get_source_duration(url)

        source_path = os.path.join(work_dir, "source.mp4")
        await tracker.update("📥 မူရင်းဗီဒီယို ဒေါင်းနေပါသည်...", force=True)
        total_dur = await download_full(url, source_path, tracker, probe_dur)

        if total_dur <= 0:
            raise RuntimeError("မူရင်းဗီဒီယို ဒေါင်းမရပါ။")

        size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(f"Full source: {total_dur:.1f}s = {fmt_time(total_dur)}, {size_mb:.0f}MB")
        await tracker.update(
            f"✅ မူရင်းရယူပြီး ({fmt_time(total_dur)}, {size_mb:.0f}MB)\n"
            f"✂️ အပိုင်းခွဲနေပါသည်...",
            force=True,
        )

        total_parts = max(1, int(np.ceil(total_dur / CHUNK_DURATION)))
        chunks = []

        for part_num in range(1, total_parts + 1):
            if job_id in CANCELLED_JOBS:
                raise Exception("cancelled")

            start = (part_num - 1) * CHUNK_DURATION
            end = min(start + CHUNK_DURATION, total_dur)
            chunk_dur = end - start

            chunk_path = os.path.join(work_dir, f"chunk_{part_num}.mp4")

            await tracker.update(
                f"✂️ အပိုင်း {part_num}/{total_parts} ခွဲနေပါသည်...\n"
                f"📊 {fmt_progress_bar(int(((part_num - 1) / total_parts) * 100))}",
                force=True,
            )

            extract_result = await asyncio.to_thread(run_cmd, [
                FFMPEG_BIN, "-y",
                "-ss", f"{start:.3f}",
                "-i", source_path,
                "-t", f"{chunk_dur:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                chunk_path,
            ], 600)

            if extract_result.returncode != 0 or not os.path.exists(chunk_path):
                raise RuntimeError(f"အပိုင်း {part_num} ဖြတ်၍မရပါ။")

            actual_dur, _, _ = probe_video_info(chunk_path)
            if actual_dur <= 0:
                raise RuntimeError(f"အပိုင်း {part_num} ကြာချိန် မရပါ။")

            logger.info(f"Chunk {part_num}/{total_parts}: {actual_dur:.1f}s")

            await tracker.update(
                f"🎧 အပိုင်း {part_num}/{total_parts} ကို Recap လုပ်နေပါသည်...\n"
                f"📊 {fmt_progress_bar(int((start / total_dur) * 100))}",
                force=True,
            )

            async with video_render_semaphore:
                recap_chunk, new_summary = await build_chunk_recap(
                    chunk_path,
                    actual_dur,
                    start,
                    start + actual_dur,
                    story_context,
                    part_num,
                    tracker,
                )

            if not recap_chunk:
                raise RuntimeError(f"အပိုင်း {part_num} Recap မရပါ။")

            chunks.append(recap_chunk)
            story_context = new_summary
            logger.info(
                f"Story context: protagonist='{story_context.get('protagonist', '')}', "
                f"{len(story_context.get('characters', []))} chars, "
                f"{len(story_context.get('summary', ''))} summary"
            )

            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

        if os.path.exists(source_path):
            try:
                os.remove(source_path)
            except Exception:
                pass

        await tracker.update("🔗 အပိုင်းတွေ ပေါင်းစပ်နေပါသည်...", force=True)
        final_out = os.path.join(work_dir, "final_recap.mp4")
        concat_final = os.path.join(work_dir, "final_concat.txt")
        with open(concat_final, "w") as f:
            for c in chunks:
                f.write(f"file '{c}'\n")

        run_cmd([
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_final,
            "-c", "copy",
            "-movflags", "+faststart",
            final_out,
        ], 300)

        final_dur, _, _ = probe_video_info(final_out)
        await tracker.update(
            f"📤 ဗီဒီယို ပို့ပေးနေပါသည်... ({fmt_time(final_dur)})\n"
            f"📊 {fmt_progress_bar(99)}",
            force=True,
        )

        parts = await split_for_telegram(final_out, work_dir)
        for i, part in enumerate(parts):
            caption = f"📹 Recap ပြီးပါပြီ။ ({fmt_time(final_dur)}) · 🛡️ Shield ဖွင့်ထားသည်"
            if len(parts) > 1:
                caption += f" - အပိုင်း {i + 1}/{len(parts)}"
            res = await tg_upload_media(
                "sendVideo",
                {"chat_id": str(chat_id), "caption": caption},
                "video",
                f"recap_{i + 1}.mp4",
                part,
                "video/mp4",
            )
            if not res.get("ok"):
                await edit_text(chat_id, status_id, "⚠️ ဗီဒီယိုပို့ရာတွင် အခက်အခဲရှိပါသည်။")
                break

        try:
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
        except Exception:
            pass

    except Exception as e:
        logger.exception("Job Error")
        await edit_text(chat_id, status_id, f"❌ အမှား: {type(e).__name__} - {str(e)[:120]}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


async def process_admin_update(update: dict):
    msg = update.get("message")
    if not msg:
        return
    chat_id = msg.get("chat", {}).get("id")
    msg_id = msg.get("message_id")
    text = (msg.get("text") or "").strip()

    if text and text.startswith("http"):
        await handle_link_job(chat_id, msg_id, text)


app = FastAPI()


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    asyncio.create_task(process_admin_update(update))
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
