import os
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import database
import tasks
from database import SessionLocal, engine

# Create tables
database.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Caramel BE API - Refactored")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints ---

@app.post("/videos", status_code=200)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a video, saves it, runs analysis synchronously,
    and returns the UUID and analysis tags.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_uuid = str(uuid.uuid4())
    video_filename = f"{file_uuid}_{file.filename}"
    video_path = UPLOAD_DIR / video_filename

    with open(video_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    new_video = database.Video(
        uuid=file_uuid,
        video_path=str(video_path),
        status='processing'
    )
    db.add(new_video)
    db.commit()
    db.refresh(new_video)

    try:
        # Run analysis synchronously
        await tasks.run_video_analysis(db, new_video)
        analysis_result = json.loads(new_video.analysis_result) if new_video.analysis_result else None

        return {
            "message": "Video upload and analysis successful.",
            "uuid": new_video.uuid,
            "analysis": analysis_result
        }
    except Exception as e:
        # The task function now re-raises the exception
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {str(e)}")

@app.post("/videos/{uuid}/thumbnail", status_code=202)
async def trigger_thumbnail_generation(
    uuid: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status not in ['analyzed', 'music_complete']:
        raise HTTPException(status_code=400, detail=f"Video is not ready for thumbnail generation. Current status: {video.status}")

    background_tasks.add_task(tasks.run_thumbnail_generation, video.id)
    return {"message": "Thumbnail generation started.", "uuid": uuid}

@app.post("/videos/{uuid}/music", status_code=202)
async def trigger_music_generation(
    uuid: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status not in ['analyzed', 'thumbnail_complete']:
        raise HTTPException(status_code=400, detail=f"Video is not ready for music generation. Current status: {video.status}")

    background_tasks.add_task(tasks.run_music_generation, video.id)
    return {"message": "Music generation started.", "uuid": uuid}


@app.get("/videos/{uuid}")
def get_video_status(uuid: str, db: Session = Depends(get_db)):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    analysis_result = None
    if video.analysis_result:
        analysis_result = json.loads(video.analysis_result)

    return {
        "uuid": video.uuid,
        "status": video.status,
        "created_at": video.created_at,
        "analysis": analysis_result,
        "files": {
            "video": f"/files/video/{video.uuid}" if video.video_path else None,
            "thumbnail": f"/files/thumbnail/{video.uuid}" if video.thumbnail_path else None,
            "music": f"/files/music/{video.uuid}" if video.music_path else None,
        }
    }

@app.get("/files/video/{uuid}")
def download_video(uuid: str, db: Session = Depends(get_db)):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video or not Path(video.video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(video.video_path)

@app.get("/files/thumbnail/{uuid}")
def download_thumbnail(uuid: str, db: Session = Depends(get_db)):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video or not video.thumbnail_path or not Path(video.thumbnail_path).exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found or not generated yet")
    return FileResponse(video.thumbnail_path)

@app.get("/files/music/{uuid}")
def download_music(uuid: str, db: Session = Depends(get_db)):
    video = db.query(database.Video).filter(database.Video.uuid == uuid).first()
    if not video or not video.music_path or not Path(video.music_path).exists():
        raise HTTPException(status_code=404, detail="Music not found or not generated yet")
    return FileResponse(video.music_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)