from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import base64
from pathlib import Path
from dotenv import load_dotenv
import uuid
import json
import google.generativeai as genai  # (영상 분석용 Gemini)
import time
import asyncio
from datetime import datetime
import threading
import contextlib  # ← 태스크 정리를 위한 suppress

# Lyria(실시간 음악)용 라이브러리
# pip install google-genai
try:
    from google import genai as genai_live
    from google.genai import types as live_types
    LYRIA_AVAILABLE = True
except Exception:
    genai_live = None
    live_types = None
    LYRIA_AVAILABLE = False

# 환경변수 로드
load_dotenv()

app = FastAPI(title="Video Analysis & Music Generation API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 로드 및 검증
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
THUMBNAIL_API_KEY = os.getenv("THUMBNAIL_API_KEY")
# Lyria용 키(없으면 GEMINI_API_KEY 재사용)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY

# 시작 시 API 키 상태 출력
print("\n" + "="*60)
print("🔑 API 키 설정 상태 확인")
print("="*60)
print(f"✅ GEMINI_API_KEY: {'설정됨' if GEMINI_API_KEY and GEMINI_API_KEY != 'None' else '❌ 미설정'}")
print(f"✅ GOOGLE_API_KEY: {'설정됨' if GOOGLE_API_KEY and GOOGLE_API_KEY != 'None' else '❌ 미설정'}")
print(f"✅ THUMBNAIL_API_KEY: {'설정됨' if THUMBNAIL_API_KEY and THUMBNAIL_API_KEY != 'None' else '❌ 미설정'}")
print(f"🎵 Lyria 라이브러리 설치: {'O' if LYRIA_AVAILABLE else 'X'}")
print("="*60 + "\n")

# Gemini 설정(영상 분석)
if GEMINI_API_KEY and GEMINI_API_KEY != "None":
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API 설정 완료\n")
else:
    print("⚠️ Gemini API 키가 없습니다.\n")

# 디렉토리 생성
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MUSIC_DIR = OUTPUT_DIR / "music"
THUMBNAIL_DIR = OUTPUT_DIR / "thumbnails"

for directory in [UPLOAD_DIR, OUTPUT_DIR, MUSIC_DIR, THUMBNAIL_DIR]:
    directory.mkdir(exist_ok=True)


def log_with_timestamp(message: str):
    """타임스탬프와 스레드 정보를 포함한 로그 출력"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = threading.current_thread().name
    print(f"[{current_time}] [{thread_id}] {message}")


# ✅ async 제거 - 단순 딕셔너리 반환
@app.get("/")
def root():
    """API 상태 확인"""
    return {
        "message": "Video Analysis & Music Generation API",
        "status": "running",
        "endpoints": {
            "GET /health": "서버 상태 확인",
            "POST /upload-video": "영상 업로드 및 전체 파이프라인 실행",
            "GET /files/video/{filename}": "업로드된 영상 다운로드",
            "GET /files/music/{filename}": "생성된 음악 다운로드",
            "GET /files/thumbnail/{filename}": "생성된 썸네일 다운로드",
            "GET /docs": "API 문서 (Swagger UI)",
            "GET /redoc": "API 문서 (ReDoc)"
        }
    }


# ✅ async 제거 - 단순 딕셔너리 반환
@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "api_keys_configured": {
            "gemini_api": bool(GEMINI_API_KEY and GEMINI_API_KEY != "None"),
            "google_live_music": bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "None" and LYRIA_AVAILABLE),
            "thumbnail_generation": bool(THUMBNAIL_API_KEY and THUMBNAIL_API_KEY != "None")
        }
    }


def analyze_audio_volume(video_path: str) -> dict:
    """영상의 소리 크기 분석"""
    try:
        log_with_timestamp("🔊 [오디오 분석] 시작")
        import subprocess

        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if 'audio' in (result.stdout or '').lower():
            audio_info = {
                "has_audio": True,
                "volume_level": "medium",
                "description": "오디오 포함"
            }
        else:
            audio_info = {
                "has_audio": False,
                "volume_level": "none",
                "description": "오디오 없음"
            }

        log_with_timestamp(f"✅ [오디오 분석] 완료: {audio_info['description']}")
        return audio_info
    except Exception as e:
        log_with_timestamp(f"⚠️ [오디오 분석] 경고: {str(e)}")
        return {
            "has_audio": False,
            "volume_level": "unknown",
            "description": "오디오 분석 불가"
        }


async def analyze_video_with_gemini(video_path: str) -> dict:
    """Gemini API로 영상 분석 (비동기 버전)"""
    try:
        log_with_timestamp("🤖 [Gemini 분석] 시작")

        if not GEMINI_API_KEY or GEMINI_API_KEY == "None":
            raise Exception("Gemini API 키가 설정되지 않았습니다")

        log_with_timestamp(f"🔑 [Gemini 분석] API 키 확인: {GEMINI_API_KEY[:20]}...")
        log_with_timestamp("📤 [Gemini 분석] 영상 업로드 시작")

        # 동기 API 호출을 별도 스레드에서 실행
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        log_with_timestamp(f"✅ [Gemini 분석] 영상 업로드 완료: {video_file.name}")

        log_with_timestamp("⏳ [Gemini 분석] 영상 처리 대기 중...")
        max_wait_time = 300  # 5분으로 증가
        wait_interval = 3
        elapsed_time = 0

        while video_file.state.name == "PROCESSING":
            if elapsed_time >= max_wait_time:
                log_with_timestamp(f"❌ [Gemini 분석] 타임아웃: {max_wait_time}초 초과")
                raise Exception(f"Gemini 파일 처리 타임아웃 ({max_wait_time}초 초과)")

            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval

            try:
                video_file = await asyncio.to_thread(genai.get_file, video_file.name)
                if elapsed_time % 15 == 0:  # 15초마다만 로그
                    log_with_timestamp(f"⏳ [Gemini 분석] 처리 중... ({elapsed_time}초 경과)")
            except Exception as e:
                log_with_timestamp(f"⚠️ [Gemini 분석] 파일 상태 확인 실패: {str(e)}")

        if video_file.state.name == "FAILED":
            raise Exception("Gemini 영상 처리 실패")

        log_with_timestamp(f"✅ [Gemini 분석] 영상 처리 완료 (소요: {elapsed_time}초)")

        model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

        prompt = """
        이 영상을 분석하여 다음 정보를 JSON 형식으로 제공해주세요:

        {
            "location": "장소 (예: 거실, 해변, 도시 거리 등)",
            "atmosphere": "분위기 (예: 평화로운, 활기찬, 고요한 등)",
            "objects": ["물체1", "물체2", "물체3"],
            "scene_description": "전체적인 장면 설명",
            "mood": "전반적인 무드 (happy, sad, calm, energetic, mysterious 중 하나)",
            "dominant_colors": ["#HEX색상1", "#HEX색상2", "#HEX색상3"],
            "mood_keywords": ["감성키워드1", "감성키워드2", "감성키워드3"]
        }

        dominant_colors: 영상에서 가장 많이 보이는 대표 색상 3가지를 HEX 코드로 (예: #FF5733, #3498DB, #2ECC71)
        mood_keywords: 색상과 분위기를 기반으로 한 감성 키워드 3개 (예: 따뜻함, 싱그러움, 차분함)

        반드시 위 JSON 형식으로만 응답해주세요.
        """

        log_with_timestamp("🔍 [Gemini 분석] AI 분석 요청 중...")
        response = await asyncio.to_thread(model.generate_content, [video_file, prompt])

        log_with_timestamp(f"📝 [Gemini 분석] AI 응답 받음")

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        gemini_result = json.loads(response_text.strip())
        log_with_timestamp("✅ [Gemini 분석] JSON 파싱 완료")

        # 오디오 분석을 별도 스레드에서 실행
        audio_info = await asyncio.to_thread(analyze_audio_volume, video_path)

        result = {
            "location": gemini_result.get("location", ""),
            "atmosphere": gemini_result.get("atmosphere", ""),
            "objects": gemini_result.get("objects", []),
            "scene_description": gemini_result.get("scene_description", ""),
            "mood": gemini_result.get("mood", "calm"),
            "dominant_colors": gemini_result.get("dominant_colors", ["#FFFFFF", "#000000", "#808080"]),
            "mood_keywords": gemini_result.get("mood_keywords", ["차분함", "자연스러움", "평온함"]),
            "audio": audio_info
        }

        log_with_timestamp("✅ [Gemini 분석] 전체 분석 완료")
        return result

    except Exception as e:
        log_with_timestamp(f"❌ [Gemini 분석] 에러: {str(e)}")
        raise Exception(f"Gemini 영상 분석 실패: {str(e)}")


# ===============================
# 🎵 음악 생성 (Google Lyria 전용) — 종료 보장 & 로그 강화
# ===============================
async def generate_music(analysis_result: dict, capture_seconds: int = 180) -> str:
    """
    Google Lyria 실시간 스트리밍으로 음악 생성 후,
    capture_seconds 동안 받은 오디오를 파일로 저장합니다.
    - 2초마다 진행 로그
    - 재생 정지 후 수신/프로그레스 태스크를 cancel+await로 명시 종료
    """
    if not LYRIA_AVAILABLE:
        raise Exception("google-genai 라이브러리가 설치되어 있지 않습니다. `pip install google-genai` 후 재시도하세요.")
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "None":
        raise Exception("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경변수가 필요합니다.")

    client = genai_live.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

    audio_buffer = bytearray()
    detected_mime = None
    total_bytes = 0

    log_with_timestamp(f"🎵 [음악 생성] Lyria 세션 준비 (캡처 {capture_seconds}s)")

    async def receive_audio(session):
        nonlocal total_bytes, detected_mime
        try:
            async for message in session.receive():
                sc = getattr(message, "server_content", None)
                if not sc:
                    continue
                chunks = getattr(sc, "audio_chunks", None) or []
                for ch in chunks:
                    data = getattr(ch, "data", None)
                    if data is None:
                        continue
                    if isinstance(data, str):
                        try:
                            b = base64.b64decode(data)
                        except Exception:
                            continue
                    elif isinstance(data, (bytes, bytearray)):
                        b = bytes(data)
                    else:
                        continue
                    audio_buffer.extend(b)
                    total_bytes += len(b)
                    mt = getattr(ch, "mime_type", None)
                    if mt and not detected_mime:
                        detected_mime = mt
                await asyncio.sleep(10**-12)
        except asyncio.CancelledError:
            # 정상 취소
            pass

    async def progress_logger():
        elapsed = 0
        while elapsed < capture_seconds:
            await asyncio.sleep(2)
            elapsed += 2
            kb = total_bytes / 1024
            if elapsed == 2:
                log_with_timestamp("🎶 [음악 생성] 생성 중… (스트리밍 수신 시작)")
            log_with_timestamp(f"⏳ [음악 생성] 수신 진행 {elapsed}/{capture_seconds}s, 누적 {kb:.1f}KB")

    # 분석 결과 기반 프롬프트
    mood = (analysis_result.get("mood") or "calm").lower()
    mood_keywords = analysis_result.get("mood_keywords", [])
    location = analysis_result.get("location", "")
    prompt_text = "minimal techno"
    if location or mood_keywords:
        prompt_text = f"minimal techno, {mood}, {' '.join(mood_keywords)} for {location}".strip()
    bpm = 90

    # 세션 열기 & 스트리밍
    try:
        async with (
            client.aio.live.music.connect(model='models/lyria-realtime-exp') as session,
            asyncio.TaskGroup() as tg,
        ):
            # 수신 & 진행 로그 태스크 등록 (핸들 저장)
            recv_task = tg.create_task(receive_audio(session))
            prog_task = tg.create_task(progress_logger())

            # 프롬프트 & 설정
            await session.set_weighted_prompts(
                prompts=[live_types.WeightedPrompt(text=prompt_text, weight=1.0)]
            )
            await session.set_music_generation_config(
                config=live_types.LiveMusicGenerationConfig(bpm=bpm, temperature=1.0)
            )

            # 재생 시작
            log_with_timestamp("▶️ [음악 생성] 재생 시작")
            await session.play()

            # 지정 시간 수신
            await asyncio.sleep(capture_seconds)

            # 재생 정지 시도
            log_with_timestamp("⏹️ [음악 생성] 재생 정지 요청")
            with contextlib.suppress(Exception):
                await session.pause()

            # === 핵심: 수신/프로그레스 태스크를 명시적으로 종료 ===
            for t in (recv_task, prog_task):
                t.cancel()
            for t in (recv_task, prog_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            log_with_timestamp("🧹 [음악 생성] 수신/프로그레스 태스크 정리 완료")

        # async with 종료 → 세션 정리됨
        log_with_timestamp("🔚 [음악 생성] 세션 종료")
    except Exception as e:
        log_with_timestamp(f"❌ [음악 생성] 세션 에러: {e}")
        raise

    if total_bytes == 0:
        raise Exception("Lyria에서 오디오 데이터를 수신하지 못했습니다. (방화벽/네트워크/WebSocket 차단 가능)")

    # mime 기반 확장자
    ext = ".mp3"
    if detected_mime:
        mt = detected_mime.lower()
        if "wav" in mt:
            ext = ".wav"
        elif "ogg" in mt or "opus" in mt:
            ext = ".ogg"
        elif "mpeg" in mt or "mp3" in mt:
            ext = ".mp3"

    # 저장 시작 로그
    kb = total_bytes / 1024
    log_with_timestamp(f"💾 [음악 생성] 파일 저장 시작 (누적 {kb:.1f}KB, mime: {detected_mime or 'unknown'})")

    music_filename = f"{uuid.uuid4()}{ext}"
    music_path = MUSIC_DIR / music_filename
    with open(music_path, "wb") as f:
        f.write(audio_buffer)

    log_with_timestamp(f"✅ [음악 생성] 저장 완료: {music_path}")
    return str(music_path)


async def generate_thumbnail(analysis_result: dict, video_path: str) -> str:
    """썸네일 생성"""
    try:
        log_with_timestamp("🖼️ [썸네일 생성] 시작")

        if not THUMBNAIL_API_KEY or THUMBNAIL_API_KEY == "None":
            raise Exception("썸네일 API 키가 설정되지 않았습니다")

        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

        location = analysis_result.get("location", "")
        atmosphere = analysis_result.get("atmosphere", "")
        objects = ", ".join(analysis_result.get("objects", [])[:3])
        mood_keywords = ", ".join(analysis_result.get("mood_keywords", []))

        prompt = f"Professional thumbnail image: {location} scene, {atmosphere} atmosphere, featuring {objects}, mood: {mood_keywords}, high quality, vibrant colors, cinematic"

        log_with_timestamp(f"📝 [썸네일 생성] 프롬프트: {prompt[:80]}...")

        headers = {
            "Authorization": f"Bearer {THUMBNAIL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }

        log_with_timestamp("📤 [썸네일 생성] API 요청 전송")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            log_with_timestamp(f"📝 [썸네일 생성] 응답 상태: {response.status_code}")

            if response.status_code != 200:
                raise Exception(f"썸네일 API 호출 실패: {response.status_code}")

            result = response.json()

            if "artifacts" not in result or len(result["artifacts"]) == 0:
                raise Exception("썸네일 API 응답에 이미지 없음")

            image_data = result["artifacts"][0]["base64"]

            thumbnail_filename = f"{uuid.uuid4()}.png"
            thumbnail_path = THUMBNAIL_DIR / thumbnail_filename

            with open(thumbnail_path, "wb") as f:
                f.write(base64.b64decode(image_data))

            log_with_timestamp(f"✅ [썸네일 생성] 완료: {thumbnail_path}")
            return str(thumbnail_path)

    except Exception as e:
        log_with_timestamp(f"❌ [썸네일 생성] 에러: {str(e)}")
        raise Exception(f"썸네일 생성 실패: {str(e)}")


@app.post("/upload-video")
async def upload_and_process_video(file: UploadFile = File(...)):
    """영상 업로드 및 전체 파이프라인 실행"""
    request_id = str(uuid.uuid4())[:8]

    try:
        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"🎬 [요청 {request_id}] 새 영상 처리 시작")
        log_with_timestamp(f"{'='*60}")

        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식")

        video_filename = f"{uuid.uuid4()}_{file.filename}"
        video_path = UPLOAD_DIR / video_filename

        log_with_timestamp(f"💾 [요청 {request_id}] 영상 저장 중: {file.filename}")
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        log_with_timestamp(f"✅ [요청 {request_id}] 영상 저장 완료")

        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"🚀 [요청 {request_id}] Gemini 분석 백그라운드로 시작")
        log_with_timestamp(f"{'='*60}")

        analysis_task = asyncio.create_task(analyze_video_with_gemini(str(video_path)))

        log_with_timestamp(f"✅ [요청 {request_id}] Gemini 태스크 생성 완료")

        # 결과 대기
        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"⏳ [요청 {request_id}] Gemini 결과 대기 (최대 330초)")
        log_with_timestamp(f"{'='*60}")

        try:
            analysis_result = await asyncio.wait_for(analysis_task, timeout=330.0)
            log_with_timestamp(f"✅ [요청 {request_id}] Gemini 분석 완료")
        except asyncio.TimeoutError:
            log_with_timestamp(f"❌ [요청 {request_id}] Gemini 타임아웃 (330초)")
            analysis_task.cancel()
            raise HTTPException(status_code=504, detail="영상 분석 시간 초과")
        except asyncio.CancelledError:
            log_with_timestamp(f"❌ [요청 {request_id}] Gemini 취소됨")
            raise HTTPException(status_code=499, detail="요청 취소")
        except Exception as e:
            log_with_timestamp(f"❌ [요청 {request_id}] Gemini 실패: {str(e)}")
            raise HTTPException(status_code=500, detail=f"영상 분석 실패: {str(e)}")

        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"⚡ [요청 {request_id}] 음악 & 썸네일 병렬 생성 (Lyria)")
        log_with_timestamp(f"{'='*60}")

        try:
            # 캡처 시간을 짧게(8s) 지정하고, 완료 후 경로를 별도 로그로 남김
            music_task = generate_music(analysis_result, capture_seconds=8)
            thumbnail_task = generate_thumbnail(analysis_result, str(video_path))

            music_path, thumbnail_path = await asyncio.gather(music_task, thumbnail_task)

            log_with_timestamp(f"🎧 [요청 {request_id}] 음악 파일 준비 완료 → {music_path}")
            log_with_timestamp(f"🖼️ [요청 {request_id}] 썸네일 파일 준비 완료 → {thumbnail_path}")
            log_with_timestamp(f"✅ [요청 {request_id}] 병렬 작업 완료")
        except Exception as e:
            log_with_timestamp(f"❌ [요청 {request_id}] 후처리 실패: {str(e)}")
            raise HTTPException(status_code=500, detail=f"후처리 실패: {str(e)}")

        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"🎉 [요청 {request_id}] 모든 작업 완료!")
        log_with_timestamp(f"{'='*60}")

        return JSONResponse({
            "success": True,
            "message": "처리 완료!",
            "data": {
                "video_analysis": {
                    "location": analysis_result["location"],
                    "atmosphere": analysis_result["atmosphere"],
                    "objects": analysis_result["objects"],
                    "scene_description": analysis_result["scene_description"],
                    "mood": analysis_result["mood"],
                    "dominant_colors": analysis_result["dominant_colors"],
                    "mood_keywords": analysis_result["mood_keywords"],
                    "audio": analysis_result["audio"]
                },
                "files": {
                    "video": f"/files/video/{video_filename}",
                    "music": f"/files/music/{Path(music_path).name}",
                    "thumbnail": f"/files/thumbnail/{Path(thumbnail_path).name}"
                }
            }
        }, media_type="application/json")  # ← 명시(선택)

    except HTTPException:
        raise
    except Exception as e:
        log_with_timestamp(f"❌ [요청 {request_id}] 예상치 못한 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")


# ✅ async 제거 - 단순 FileResponse 반환
@app.get("/files/video/{filename}")
def get_video(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# ✅ async 제거 - 단순 FileResponse 반환
@app.get("/files/music/{filename}")
def get_music(filename: str):
    file_path = MUSIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# ✅ async 제거 - 단순 FileResponse 반환
@app.get("/files/thumbnail/{filename}")
def get_thumbnail(filename: str):
    file_path = THUMBNAIL_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    log_with_timestamp("🚀 서버 시작")
    uvicorn.run(app, host="0.0.0.0", port=8000)
