from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import uuid
import json
import google.generativeai as genai  # (영상 분석용 Gemini)
import asyncio
from datetime import datetime
import threading
import contextlib  # 태스크 정리를 위한 suppress
import re
import time
import math
import traceback

# Lyria(실시간 음악) & Imagen(이미지)용 라이브러리 (google-genai)
# pip install google-genai
try:
    from google import genai as genai_live
    from google.genai import types as live_types
    LYRIA_AVAILABLE = True
except Exception:
    genai_live = None
    live_types = None
    LYRIA_AVAILABLE = False

# PIL은 선택(Imagen이 PIL Image 객체를 반환) — 없으면 bytes로 저장 시도
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

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
# Lyria/Imagen용 키 (없으면 GEMINI_API_KEY 재사용)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY

# 시작 시 API 키 상태 출력
print("\n" + "="*60)
print("🔑 API 키 설정 상태 확인")
print("="*60)
print(f"✅ GEMINI_API_KEY: {'설정됨' if GEMINI_API_KEY and GEMINI_API_KEY != 'None' else '❌ 미설정'}")
print(f"✅ GOOGLE_API_KEY: {'설정됨' if GOOGLE_API_KEY and GOOGLE_API_KEY != 'None' else '❌ 미설정'}")
print(f"🎵 Lyria 라이브러리 설치: {'O' if LYRIA_AVAILABLE else 'X'}")
print(f"🖼️ PIL 설치: {'O' if PIL_AVAILABLE else 'X'}")
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


@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "api_keys_configured": {
            "gemini_api": bool(GEMINI_API_KEY and GEMINI_API_KEY != "None"),
            "google_live_music": bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "None" and LYRIA_AVAILABLE),
            "google_imagen": bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "None")
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


def _normalize_list(raw, max_len=20) -> list:
    """리스트/문자열을 최대 max_len 아이템의 문자열 리스트로 정규화"""
    if raw is None:
        items = []
    elif isinstance(raw, list):
        items = []
        for x in raw:
            if not x:
                continue
            if isinstance(x, str):
                # 쉼표/슬래시/파이프 분리도 허용
                items.extend([t.strip() for t in re.split(r"[,/|]", x) if t.strip()])
            else:
                items.append(str(x))
    elif isinstance(raw, str):
        items = [t.strip() for t in re.split(r"[,/|]", raw) if t.strip()]
    else:
        items = [str(raw)]

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for t in items:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_len:
            break
    return uniq


def _normalize_keywords(raw) -> list:
    """키워드 최대 20개로 정규화"""
    return _normalize_list(raw, max_len=20)


def _auto_expand_keywords(base_keywords: list, mood: str, location: str, atmosphere: str,
                          objects: list, dominant_colors: list) -> list:
    """키워드가 20개에 못 미치면 관련 필드로 자동 보강"""
    pool = list(base_keywords)

    # 분위기/무드 기반 제안
    mood_map = {
        "calm": ["차분함", "잔잔함", "미니멀", "여유로운", "편안한", "웜톤", "소프트", "lofi", "ambient"],
        "happy": ["밝음", "경쾌함", "명랑함", "팝", "업비트", "신나는", "비비드", "하이라이트", "리드미컬"],
        "energetic": ["에너지", "업템포", "드라이브", "리듬", "비트감", "댄서블", "다이내믹", "강렬함", "상승감"],
        "sad": ["쓸쓸함", "서정적", "잔향", "서브듀드", "다운템포", "블루톤", "감성적", "우울한", "슬로우"],
        "mysterious": ["미스테리", "몽환적", "신비로운", "에코", "싱코페이션", "어두운 톤", "미드나잇", "드론"]
    }
    for k in mood_map.get((mood or "").lower(), []):
        if k not in pool:
            pool.append(k)

    # 장소/분위기 보강
    for t in [location, atmosphere]:
        if t and t not in pool:
            pool.append(t)

    # 오브젝트 상위 5개
    for obj in (objects or [])[:5]:
        if obj and obj not in pool:
            pool.append(obj)

    # 색상 키워드 러프 매핑
    color_map = {
        "red": ["따뜻한", "열정적"], "green": ["싱그러운", "내추럴"], "blue": ["차분한", "시원한"],
        "yellow": ["선명한", "밝은"], "orange": ["비비드", "따뜻함"], "purple": ["몽환적", "감성적"],
        "pink": ["부드러운", "러블리"], "black": ["모던", "미니멀"], "white": ["클린", "미니멀"],
        "gray": ["모노톤", "세련된"], "brown": ["빈티지", "따뜻한"]
    }

    def color_to_keywords(hexcode: str):
        if not isinstance(hexcode, str):
            return []
        name_hits = []
        lower = hexcode.lower()
        for name, kws in color_map.items():
            if name in lower:
                name_hits.extend(kws)
        return name_hits

    for dc in (dominant_colors or [])[:3]:
        for k in color_to_keywords(dc):
            if k not in pool:
                pool.append(k)

    common = ["시네마틱", "브랜드무드", "현대적", "클린", "텍스쳐", "감도", "컨트라스트", "하이라이트",
              "네온", "빈티지", "아날로그", "모션", "리드미컬", "그루브"]
    for k in common:
        if k not in pool:
            pool.append(k)

    # 중복 제거 후 20개 제한
    seen = set()
    out = []
    for t in pool:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 20:
            break
    return out


async def analyze_video_with_gemini(video_path: str) -> dict:
    """Gemini API로 영상 분석 (비동기 버전)"""
    try:
        log_with_timestamp("🤖 [Gemini 분석] 시작")

        if not GEMINI_API_KEY or GEMINI_API_KEY == "None":
            raise Exception("Gemini API 키가 설정되지 않았습니다")

        log_with_timestamp(f"🔑 [Gemini 분석] API 키 확인: {GEMINI_API_KEY[:3]}...")
        log_with_timestamp("📤 [Gemini 분석] 영상 업로드 시작")

        # 동기 API 호출을 별도 스레드에서 실행
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        log_with_timestamp(f"✅ [Gemini 분석] 영상 업로드 완료: {video_file.name}")

        log_with_timestamp("⏳ [Gemini 분석] 영상 처리 대기 중...")
        max_wait_time = 300  # 5분
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
                if elapsed_time % 15 == 0:  # 15초마다 로그
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
            "objects": ["장면 내 핵심 오브젝트 15~20개 (사람, 사물, 간판, 장치, 소품 등)"],
            "scene_description": "전체적인 장면 설명",
            "mood": "전반적인 무드 (happy, sad, calm, energetic, mysterious 중 하나)",
            "dominant_colors": ["#HEX색상1", "#HEX색상2", "#HEX색상3"],
            "mood_keywords": ["해당 장면/무드/색감/오브젝트/스타일을 설명하는 한국어 키워드 15~20개"]
        }

        규칙:
        - 반드시 위 JSON 형식으로만 응답 (추가 텍스트 금지)
        - objects: 최대한 다양하게 15~20개, 배열로 반환
        - mood_keywords: 한국어로 15~20개, 배열로 반환
        - dominant_colors: 가능하면 실제 장면의 대표 색상을 HEX로 답변
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

        # 필드 추출 + 정규화
        location = gemini_result.get("location", "")
        atmosphere = gemini_result.get("atmosphere", "")
        objects = _normalize_list(gemini_result.get("objects"), max_len=20)
        scene_description = gemini_result.get("scene_description", "")
        mood = gemini_result.get("mood", "calm")
        dominant_colors = _normalize_list(gemini_result.get("dominant_colors"), max_len=3)

        # 키워드 정규화 + 20개 보장
        raw_keywords = gemini_result.get("mood_keywords")
        normalized_keywords = _normalize_keywords(raw_keywords)
        if len(normalized_keywords) < 20:
            normalized_keywords = _auto_expand_keywords(
                normalized_keywords, mood, location, atmosphere, objects, dominant_colors
            )

        result = {
            "location": location,
            "atmosphere": atmosphere,
            "objects": objects,  # ⇐ 최대 20개
            "scene_description": scene_description,
            "mood": mood,
            "dominant_colors": dominant_colors,
            "mood_keywords": normalized_keywords,  # ⇐ 최대 20개
            "audio": audio_info
        }

        log_with_timestamp(f"✅ [Gemini 분석] 전체 분석 완료 (objects {len(objects)}개, keywords {len(normalized_keywords)}개)")
        return result

    except Exception as e:
        log_with_timestamp(f"❌ [Gemini 분석] 에러: {str(e)}")
        raise Exception(f"Gemini 영상 분석 실패: {str(e)}")


# ===============================
# 🎵 음악 생성 (Google Lyria 전용) — 종료 보장 & 로그 강화
# ===============================
async def generate_music(analysis_result: dict, capture_seconds: int = 8) -> str:
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

    try:
        async with (
            client.aio.live.music.connect(model='models/lyria-realtime-exp') as session,
            asyncio.TaskGroup() as tg,
        ):
            recv_task = tg.create_task(receive_audio(session))
            prog_task = tg.create_task(progress_logger())

            await session.set_weighted_prompts(
                prompts=[live_types.WeightedPrompt(text=prompt_text, weight=1.0)]
            )
            await session.set_music_generation_config(
                config=live_types.LiveMusicGenerationConfig(bpm=bpm, temperature=1.0)
            )

            log_with_timestamp("▶️ [음악 생성] 재생 시작")
            await session.play()

            await asyncio.sleep(capture_seconds)

            log_with_timestamp("⏹️ [음악 생성] 재생 정지 요청")
            with contextlib.suppress(Exception):
                await session.pause()

            for t in (recv_task, prog_task):
                t.cancel()
            for t in (recv_task, prog_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            log_with_timestamp("🧹 [음악 생성] 수신/프로그레스 태스크 정리 완료")

        log_with_timestamp("🔚 [음악 생성] 세션 종료")
    except Exception as e:
        log_with_timestamp(f"❌ [음악 생성] 세션 에러: {e}")
        raise

    if total_bytes == 0:
        raise Exception("Lyria에서 오디오 데이터를 수신하지 못했습니다. (네트워크/WebSocket 차단 가능)")

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

    kb = total_bytes / 1024
    log_with_timestamp(f"💾 [음악 생성] 파일 저장 시작 (누적 {kb:.1f}KB, mime: {detected_mime or 'unknown'})")

    music_filename = f"{uuid.uuid4()}{ext}"
    music_path = MUSIC_DIR / music_filename
    with open(music_path, "wb") as f:
        f.write(audio_buffer)

    log_with_timestamp(f"✅ [음악 생성] 저장 완료: {music_path}")
    return str(music_path)


# ===============================
# 🖼️ 썸네일 생성 (Google Imagen) - 429 재시도 & graceful degrade
# ===============================
async def generate_thumbnail(analysis_result: dict, video_path: str) -> str:
    """
    Google Imagen으로 썸네일 1장 생성하여 PNG로 저장.
    429 등 오류 발생 시 최대 3회 (1s→2s→4s) 재시도.
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "None":
        raise Exception("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경변수가 필요합니다.")
    if genai_live is None:
        raise Exception("google-genai 라이브러리가 설치되어 있지 않습니다. `pip install google-genai` 후 재시도하세요.")

    # 프롬프트 구성
    location = analysis_result.get("location", "")
    atmosphere = analysis_result.get("atmosphere", "")
    objects = ", ".join(analysis_result.get("objects", [])[:20])
    mood_keywords = ", ".join(analysis_result.get("mood_keywords", []))

    prompt = (
        f"Professional thumbnail image: {location} scene, {atmosphere} atmosphere, "
        f"featuring {objects}, mood: {mood_keywords}, high quality, vibrant colors, cinematic, 1:1 aspect"
    ).strip()

    log_with_timestamp("🖼️ [썸네일 생성] 시작 (Imagen)")
    log_with_timestamp(f"📝 [썸네일 생성] 프롬프트: {prompt[:100]}...")

    def _gen_and_save_with_retry() -> str:
        client = genai_live.Client(api_key=GOOGLE_API_KEY)
        attempts = 3
        backoff = 1.0
        last_err = None

        for i in range(1, attempts + 1):
            try:
                log_with_timestamp(f"📤 [썸네일 생성] API 요청 전송 (시도 {i}/{attempts})")
                resp = client.models.generate_images(
                    model='imagen-4.0-generate-001',
                    prompt=prompt,
                    config=live_types.GenerateImagesConfig(number_of_images=1),
                )
                if not getattr(resp, "generated_images", None):
                    raise Exception("Imagen 응답에 이미지가 없습니다.")
                gi = resp.generated_images[0]

                # 파일 경로
                thumbnail_filename = f"{uuid.uuid4()}.png"
                thumbnail_path = THUMBNAIL_DIR / thumbnail_filename

                # 1) PIL Image 객체가 있으면 그대로 저장
                img_obj = getattr(gi, "image", None)
                if img_obj is not None and PIL_AVAILABLE:
                    try:
                        img_obj.save(thumbnail_path)
                        return str(thumbnail_path)
                    except Exception:
                        pass  # 아래 bytes 경로로 폴백

                # 2) bytes 속성이 있으면 그대로 저장
                raw = getattr(gi, "image_bytes", None) or getattr(gi, "bytes", None) or getattr(gi, "data", None)
                if raw:
                    if isinstance(raw, str):
                        raw = base64.b64decode(raw)
                    with open(thumbnail_path, "wb") as f:
                        f.write(raw)
                    return str(thumbnail_path)

                raise Exception("이미지를 저장할 수 없습니다. PIL 설치를 권장합니다: pip install pillow")

            except Exception as e:
                last_err = e
                msg = str(e)
                # 429 또는 Rate limit 추정 시 백오프 재시도
                if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                    log_with_timestamp(f"⚠️ [썸네일 생성] 레이트 리밋/쿼터 이슈로 재시도 예정: {msg}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # 기타 에러는 즉시 중단
                log_with_timestamp(f"❌ [썸네일 생성] 에러: {msg}")
                raise

        # 모든 시도 실패
        raise last_err or Exception("Imagen 생성 실패(알 수 없는 오류)")

    # 동기 호출을 별도 스레드에서
    try:
        thumbnail_path = await asyncio.to_thread(_gen_and_save_with_retry)
        log_with_timestamp(f"✅ [썸네일 생성] 완료: {thumbnail_path}")
        return thumbnail_path
    except Exception as e:
        # 여기서 예외를 올려서 전체 파이프라인을 멈추지 않게,
        # 상위에서 graceful degrade 처리하도록 에러 텍스트를 래핑해서 다시 raise
        raise Exception(f"썸네일 생성 실패: {e}")


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
        log_with_timestamp(f"⚡ [요청 {request_id}] 음악 & 썸네일 병렬 생성 (Lyria/Imagen)")
        log_with_timestamp(f"{'='*60}")

        # 썸네일 실패해도 전체 실패하지 않도록 예외 수집
        music_task = generate_music(analysis_result, capture_seconds=8)
        thumbnail_task = generate_thumbnail(analysis_result, str(video_path))
        music_result, thumbnail_result = await asyncio.gather(
            music_task, thumbnail_task, return_exceptions=True
        )

        # 음악 결과 처리
        if isinstance(music_result, Exception):
            log_with_timestamp(f"❌ [요청 {request_id}] 음악 생성 실패: {music_result}")
            # 음악은 핵심 리소스라 실패 시 500 반환
            raise HTTPException(status_code=500, detail=f"음악 생성 실패: {str(music_result)}")
        else:
            music_path = music_result
            log_with_timestamp(f"🎧 [요청 {request_id}] 음악 파일 준비 완료 → {music_path}")

        # 썸네일 결과 처리 (실패해도 응답은 반환)
        thumbnail_path = None
        thumbnail_error = None
        if isinstance(thumbnail_result, Exception):
            thumbnail_error = str(thumbnail_result)
            log_with_timestamp(f"⚠️ [요청 {request_id}] 썸네일 생성 실패(무시하고 진행): {thumbnail_error}")
        else:
            thumbnail_path = thumbnail_result
            log_with_timestamp(f"🖼️ [요청 {request_id}] 썸네일 파일 준비 완료 → {thumbnail_path}")

        log_with_timestamp(f"✅ [요청 {request_id}] 병렬 작업 처리 완료")

        log_with_timestamp(f"{'='*60}")
        log_with_timestamp(f"🎉 [요청 {request_id}] 모든 작업 완료!")
        log_with_timestamp(f"{'='*60}")

        # 응답 구성
        resp = {
            "success": True,
            "message": "처리 완료!",
            "data": {
                "video_analysis": {
                    "location": analysis_result["location"],
                    "atmosphere": analysis_result["atmosphere"],
                    "objects": analysis_result["objects"],  # 최대 20개
                    "scene_description": analysis_result["scene_description"],
                    "mood": analysis_result["mood"],
                    "dominant_colors": analysis_result["dominant_colors"],
                    "mood_keywords": analysis_result["mood_keywords"],  # 최대 20개
                    "audio": analysis_result["audio"]
                },
                "files": {
                    "video": f"/files/video/{video_filename}",
                    "music": f"/files/music/{Path(music_path).name}",
                    "thumbnail": f"/files/thumbnail/{Path(thumbnail_path).name}" if thumbnail_path else None
                }
            }
        }

        if thumbnail_error:
            resp["data"]["thumbnail_error"] = thumbnail_error

        return JSONResponse(resp, media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        log_with_timestamp(f"❌ [요청 {request_id}] 예상치 못한 에러: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")


@app.get("/files/video/{filename}")
def get_video(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


@app.get("/files/music/{filename}")
def get_music(filename: str):
    file_path = MUSIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


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
