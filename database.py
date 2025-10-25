
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

# Fallback to user-provided credentials if .env is not set
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root_1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "caramel_be_db")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)
    video_path = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default='processing')
    analysis_result = Column(JSON, nullable=True)
    thumbnail_path = Column(String(255), nullable=True)
    music_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    # Create database if it does not exist
    from sqlalchemy_utils import database_exists, create_database
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Database '{DB_NAME}' created.")
    else:
        print(f"Database '{DB_NAME}' already exists.")

    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Tables created.")

if __name__ == "__main__":
    # This allows creating the DB and tables by running `python database.py`
    init_db()
