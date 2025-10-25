import os
import uuid
import asyncio
import httpx
from moviepy.editor import AudioFileClip, ColorClip

MUBERT_API = "https://api.mubert.com/v2"
DURATION_SEC = 15          # 생성 음악 길이(초) – 짧게 테스트
MODE = "chill"             # 무드(예: chill, ambient, techno 등)
TAGS = "minimal, soft"     # 태그(쉼표로 구분)
APPLICATION = "quick_mp4_demo"  # Mubert가 요구하는 필드

API_KEY = os.getenv("MUSIC_API_KEY")

if not API_KEY:
    raise SystemExit("환경변수 MUSIC_API_KEY가 없습니다. setx/export로 먼저 설정하세요.")

async def request_track(client: httpx.AsyncClient) -> str:
    """
    트랙 생성을 요청하고 task_id 또는 download_link를 반환.
    download_link가 바로 오면 그대로 반환, 아니면 task_id 반환.
    """
    payload = {
        "method": "RecordTrack",
        "params": {
            "license": "personal",
            "token": API_KEY,
            "mode": MODE,
            "duration": DURATION_SEC,
            "tags": TAGS,
            "application": APPLICATION
        }
    }
    r = await client.post(f"{MUBERT_API}/RecordTrack", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"Mubert 오류: {data['error']}")

    d = data.get("data", {})
    # 바로 다운로드 링크가 오는 경우
    if "download_link" in d:
        return d["download_link"]

    # task 방식
    tasks = d.get("tasks", [])
    if not tasks:
        raise RuntimeError(f"예상치 못한 응답 형식: {data}")
    return tasks[0]  # task_id

async def poll_download_link(client: httpx.AsyncClient, task_id: str) -> str:
    """task_id로 GetTrack을 폴링해 download_link 획득"""
    for _ in range(30):  # 최대 30회(약 90초)
        payload = {
            "method": "GetTrack",
            "params": {
                "token": API_KEY,
                "task_id": task_id
            }
        }
        r = await client.post(f"{MUBERT_API}/GetTrack", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        d = data.get("data", {})
        if "download_link" in d:
            return d["download_link"]
        await asyncio.sleep(3)

    raise TimeoutError("트랙 생성 대기 시간 초과")

async def download_mp3(client: httpx.AsyncClient, url: str, out_path: str):
    """MP3 파일 다운로드"""
    r = await client.get(url, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def audio_to_mp4(mp3_path: str, mp4_path: str):
    """
    검은 화면(ColorClip) + 오디오를 합쳐 MP4 저장.
    moviepy가 내부적으로 ffmpeg를 사용합니다.
    """
    audio = AudioFileClip(mp3_path)
    # 오디오 길이에 맞춘 검은 화면 (1280x720)
    video = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=audio.duration)
    video = video.set_audio(audio)
    # codec='libx264' / audio_codec='aac'가 호환성 좋음
    video.write_videofile(mp4_path, fps=24, codec="libx264", audio_codec="aac")

async def main():
    mp3_name = f"{uuid.uuid4()}.mp3"
    mp4_name = f"{uuid.uuid4()}.mp4"

    async with httpx.AsyncClient() as client:
        res = await request_track(client)
        if res.startswith("http"):
            download_link = res
        else:
            # task_id를 받은 경우 폴링
            download_link = await poll_download_link(client, res)

        await download_mp3(client, download_link, mp3_name)
        print(f"✅ MP3 저장: {mp3_name}")

    # MP4로 변환
    audio_to_mp4(mp3_name, mp4_name)
    print(f"🎬 MP4 완료: {mp4_name}")

if __name__ == "__main__":
    asyncio.run(main())
