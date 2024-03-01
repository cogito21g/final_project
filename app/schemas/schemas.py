from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, time

from pydantic import BaseModel, field_validator, EmailStr
from pydantic_core.core_schema import FieldValidationInfo

class UserBase(BaseModel):
    email: EmailStr
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password1: str
    password2: str
    
    @field_validator('email', 'password1', 'password2')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @field_validator('password2')
    def passwords_match(cls, v, info: FieldValidationInfo):
        if 'password1' in info.data and v != info.data['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str
    email: str

class TokenPayload(BaseModel):
    sub: Optional[str] = None

class User(UserBase):
    user_id: int
    uploads: List["Upload"] = []

    class Config:
        orm_mode = True

class UploadCreate(BaseModel):
    name: str
    date: datetime
    is_realtime: Optional[bool] = None
    user_id: int

class Upload(UploadCreate):
    upload_id: int
    videos: List["Video"] = []

    class Config:
        orm_mode = True

# Pydantic 스키마 - Video
class VideoCreate(BaseModel):
    video_url: str
    upload_id: int

class Video(VideoCreate):
    video_id: int
    frames: List["Frame"] = []

    class Config:
        orm_mode = True

# Pydantic 스키마 - Frame
class FrameCreate(BaseModel):
    frame_url: str
    time_stamp: time
    score: float
    video_id: int

class Frame(FrameCreate):
    frame_id: int

    class Config:
        orm_mode = True
