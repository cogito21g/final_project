from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Time
from sqlalchemy.orm import relationship, declarative_base, Session
from db.database import Base


class User(Base):
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    
    uploads = relationship("Upload", back_populates="user")

class Upload(Base):
    __tablename__ = "upload"

    upload_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    is_realtime = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    
    user = relationship("User", back_populates="uploads")
    videos = relationship("Video", back_populates="upload", cascade="all, delete-orphan")

class Video(Base):
    __tablename__ = "video"

    video_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    video_url = Column(String(255), nullable=False)
    upload_id = Column(Integer, ForeignKey("upload.upload_id"), nullable=False)
    
    upload = relationship("Upload", back_populates="videos")
    frames = relationship("Frame", back_populates="video")

class Frame(Base):
    __tablename__ = "frame"

    frame_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    frame_url = Column(String(255), nullable=False)
    time_stamp = Column(Time, nullable=False)
    score = Column(Float, nullable=False)
    video_id = Column(Integer, ForeignKey("video.video_id"), nullable=False)
    
    video = relationship("Video", back_populates="frames")
