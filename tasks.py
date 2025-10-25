import os
import httpx
import base64
from pathlib import Path
from dotenv import load_dotenv
import uuid
import json
import google.generativeai as genai
import asyncio
from datetime import datetime
import threading
import contextlib

from sqlalchemy.orm import Session
from database import Video, SessionLocal

# Lyria-related imports
try:
    from google import genai as genai_live
    from google.genai import types as live_types
    LYRIA_AVAILABLE = True
except ImportError:
    genai_live = None
    live_types = None
    LYRIA_AVAILABLE = False

load_dotenv()

# --- Constants and API Key Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
THUMBNAIL_API_KEY = os.getenv("THUMBNAIL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MUSIC_DIR = OUTPUT_DIR / "music"
THUMBNAIL_DIR = OUTPUT_DIR / "thumbnails"

if GEMINI_API_KEY and GEMINI_API_KEY != "None":
    genai.configure(api_key=GEMINI_API_KEY)

# --- Utility Functions ---
def log_with_timestamp(message: str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = threading.current_thread().name
    print(f"[{current_time}] [{thread_id}] {message}")

def analyze_audio_volume(video_path: str) -> dict:
    try:
        log_with_timestamp("🔊 [Audio Analysis] Started")
        import subprocess
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if 'audio' in (result.stdout or '').lower():
            return {"has_audio": True, "volume_level": "medium", "description": "Audio present"}
        else:
            return {"has_audio": False, "volume_level": "none", "description": "No audio"}
    except Exception as e:
        log_with_timestamp(f"⚠️ [Audio Analysis] Warning: {str(e)}")
        return {"has_audio": False, "volume_level": "unknown", "description": "Audio analysis failed"}

# --- Background Task Functions ---

async def run_video_analysis(db: Session, video: Video):
    """Runs video analysis and updates the database directly."""
    if not video:
        log_with_timestamp(f"[Analysis] Invalid video object passed.")
        return

    try:
        log_with_timestamp(f"🤖 [Analysis] Task started for video ID: {video.id}")
        video.status = 'analyzing'
        db.commit()

        video_path = video.video_path
        if not GEMINI_API_KEY or GEMINI_API_KEY == "None":
            raise Exception("Gemini API key is not configured.")

        # (Analysis logic...)
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        max_wait_time = 300
        elapsed_time = 0
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(5)
            elapsed_time += 5
            if elapsed_time > max_wait_time:
                raise Exception("Gemini processing timed out.")
            video_file = await asyncio.to_thread(genai.get_file, name=video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception("Gemini video processing failed.")

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = '''Analyze this video and provide location, atmosphere, objects, scene_description, mood, dominant_colors, and mood_keywords in JSON format.'''
        response = await asyncio.to_thread(model.generate_content, [video_file, prompt])
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        gemini_result = json.loads(response_text)
        audio_info = await asyncio.to_thread(analyze_audio_volume, video_path)

        # Update DB
        video.analysis_result = json.dumps({**gemini_result, "audio": audio_info})
        video.status = 'analyzed'
        db.commit()
        db.refresh(video) # Refresh the object to get the updated state
        log_with_timestamp(f"✅ [Analysis] Task complete for video ID: {video.id}")

    except Exception as e:
        log_with_timestamp(f"❌ [Analysis] Error for video ID {video.id}: {e}")
        video.status = 'analysis_failed'
        db.commit()
        raise e # Re-raise the exception to be caught by the endpoint

async def run_thumbnail_generation(video_id: int):
    db = SessionLocal()
    video = db.query(Video).filter(Video.id == video_id).first()
    if not (video and video.analysis_result and video.status in ['analyzed', 'music_complete']):
        log_with_timestamp(f"[Thumbnail] Video {video_id} not ready for thumbnail generation.")
        db.close()
        return

    try:
        log_with_timestamp(f"🖼️ [Thumbnail] Task started for video ID: {video_id}")
        video.status = 'generating_thumbnail'
        db.commit()

        analysis_result = json.loads(video.analysis_result)

        if not THUMBNAIL_API_KEY or THUMBNAIL_API_KEY == "None":
            raise Exception("Thumbnail API key is not configured.")

        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        prompt = (
            f"Professional thumbnail: {analysis_result.get('location', '')} scene, "
            f"{analysis_result.get('atmosphere', '')} atmosphere, featuring {analysis_result.get('objects', [])}, "
            f"mood: {', '.join(analysis_result.get('mood_keywords', []))}, high quality, cinematic"
        )

        headers = {"Authorization": f"Bearer {THUMBNAIL_API_KEY}", "Content-Type": "application/json"}
        payload = {"text_prompts": [{"text": prompt}], "samples": 1, "steps": 30}

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            image_data = response.json()["artifacts"][0]["base64"]

        thumbnail_filename = f"{uuid.uuid4()}.png"
        thumbnail_path = THUMBNAIL_DIR / thumbnail_filename
        with open(thumbnail_path, "wb") as f:
            f.write(base64.b64decode(image_data))

        video.thumbnail_path = str(thumbnail_path)
        video.status = 'thumbnail_complete' if video.status == 'analyzed' else 'completed'
        db.commit()
        log_with_timestamp(f"✅ [Thumbnail] Task complete for video ID: {video_id}")

    except Exception as e:
        log_with_timestamp(f"❌ [Thumbnail] Error for video ID {video_id}: {e}")
        video.status = 'thumbnail_failed'
        db.commit()
    finally:
        db.close()

async def run_music_generation(video_id: int, capture_seconds: int = 30):
    db = SessionLocal()
    video = db.query(Video).filter(Video.id == video_id).first()
    if not (video and video.analysis_result and video.status in ['analyzed', 'thumbnail_complete']):
        log_with_timestamp(f"[Music] Video {video_id} not ready for music generation.")
        db.close()
        return

    try:
        log_with_timestamp(f"🎵 [Music] Task started for video ID: {video_id}")
        video.status = 'generating_music'
        db.commit()

        analysis_result = json.loads(video.analysis_result)

        if not LYRIA_AVAILABLE:
            raise Exception("Lyria library (google-genai) is not installed.")
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "None":
            raise Exception("GOOGLE_API_KEY is required for music generation.")

        client = genai_live.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})
        audio_buffer = bytearray()
        detected_mime = None

        async def receive_audio(session):
            nonlocal detected_mime
            async for message in session.receive():
                if sc := getattr(message, "server_content", None):
                    for ch in getattr(sc, "audio_chunks", []):
                        if data := getattr(ch, "data", None):
                            audio_buffer.extend(base64.b64decode(data) if isinstance(data, str) else bytes(data))
                            if mt := getattr(ch, "mime_type", None):
                                detected_mime = mt

        mood = analysis_result.get("mood", "calm").lower()
        prompt_text = f"minimal techno, {mood}, {' '.join(analysis_result.get("mood_keywords", []))}"

        try:
            async with client.aio.live.music.connect(model='models/lyria-realtime-exp') as session:
                recv_task = asyncio.create_task(receive_audio(session))
                await session.set_weighted_prompts(prompts=[live_types.WeightedPrompt(text=prompt_text, weight=1.0)])
                await session.play()
                await asyncio.sleep(capture_seconds)
                await session.pause()
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task

        except Exception as e:
            log_with_timestamp(f"❌ [Music Gen] Session error: {e}")
            raise

        if not audio_buffer:
            raise Exception("No audio data received from Lyria.")

        ext = ".mp3"
        if detected_mime and "wav" in detected_mime.lower():
            ext = ".wav"

        music_filename = f"{uuid.uuid4()}{ext}"
        music_path = MUSIC_DIR / music_filename
        with open(music_path, "wb") as f:
            f.write(audio_buffer)

        video.music_path = str(music_path)
        video.status = 'music_complete' if video.status == 'analyzed' else 'completed'
        db.commit()
        log_with_timestamp(f"✅ [Music] Task complete for video ID: {video_id}")

    except Exception as e:
        log_with_timestamp(f"❌ [Music] Error for video ID {video_id}: {e}")
        video.status = 'music_failed'
        db.commit()
    finally:
        db.close()