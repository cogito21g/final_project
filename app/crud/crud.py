from datetime import timedelta, datetime, date
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
from passlib.context import CryptContext
from models import models
from schemas.schemas import UserCreate, UploadCreate, VideoCreate, FrameCreate, Complete

from core.security import get_password_hash, verify_password
from models.models import User

## User
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password1)
    db_user = models.User(email=user.email, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_existing_user(db: Session, user_create: UserCreate):
    return db.query(models.User).filter(
        (models.User.email == user_create.email)
    ).first()
    
def authenticate(db: Session, *, email: str, password: str):
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user

def is_active(user: User) -> bool:
    return user.is_active

## Upload
def create_upload(db: Session, upload: UploadCreate):
    db_upload = models.Upload(**upload.dict())
    db.add(db_upload)
    db.commit()
    db.refresh(db_upload)
    return db_upload

def delete_upload(db: Session, upload_id: int):
    db_upload = db.query(models.Upload).filter(models.Upload.upload_id == upload_id).first()
    
    if db_upload:
        db.delete(db_upload)
        db.commit()
        return True
    return False

def get_upload(db: Session, upload_id: int):
    return db.query(models.Upload).filter(models.Upload.upload_id==upload_id).first()

def get_upload_id(db: Session, user_id: int, name: str, date: datetime,):
    return db.query(models.Upload).filter(
        (models.Upload.user_id == user_id) &
        (models.Upload.name == name) &
        (models.Upload.date == date)
        ).all()

def get_uploads(db: Session, user_id: int):
    return db.query(models.Upload).filter(
        models.Upload.user_id == user_id).order_by(models.Upload.upload_id.desc()).all()
    
def get_upload_by_name(db: Session, name: str):
    return db.query(models.Upload).filter(models.Upload.name == name).first()

## Video
def create_video(db: Session, video: VideoCreate):
    db_video = models.Video(**video.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def get_video(db: Session, upload_id: int):
    return db.query(models.Video).filter(
        models.Video.upload_id == upload_id).first()

## Frame
def create_frame(db: Session, frame: FrameCreate):
    db_frame = models.Frame(**frame.dict())
    db.add(db_frame)
    db.commit()
    db.refresh(db_frame)
    return db_frame

def get_frame(db: Session, frame_id: int):
    return db.query(models.Frame).filter(models.Frame.frame_id == frame_id).first()

def get_frames(db: Session, video_id: int):
    return db.query(models.Frame).filter(models.Frame.video_id == video_id).all()

def get_frames_with_highest_score(db: Session, video_id: int):
    
    subquery = (
        db.query(
            models.Frame.video_id,
            models.Frame.time_stamp,
            func.max(models.Frame.score).label('max_score')
        )
        .group_by(models.Frame.video_id, models.Frame.time_stamp)
        .subquery()
    )
    
    subq_alias = aliased(subquery)

    frames = (
        db.query(models.Frame)
        .join(
            subq_alias,
            (models.Frame.video_id == subq_alias.c.video_id) &
            (models.Frame.time_stamp == subq_alias.c.time_stamp) &
            (models.Frame.score == subq_alias.c.max_score)
        )
        .filter(models.Frame.video_id == video_id)
        .all()
    )

    return frames

def create_complete(db: Session, complete: Complete):
    db_complete = models.Complete(**complete.dict())
    db.add(db_complete)
    db.commit()
    db.refresh(db_complete)
    return db_complete

def get_complete(db: Session, upload_id: int):
    return db.query(models.Complete).filter(models.Complete.upload_id == upload_id).first()

def update_complete_status(db: Session, upload_id: int):
    
    complete_record = db.query(models.Complete).filter(models.Complete.upload_id == upload_id).first()

    if complete_record and not complete_record.completed:
        complete_record.completed = True
        db.commit()