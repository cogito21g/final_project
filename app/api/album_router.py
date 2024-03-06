from datetime import timedelta, datetime, date
from typing import Optional
import ast

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Cookie, Query
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from core.config import get_settings
from db.database import get_db, db_engine
from models import models
from crud import crud
from crud.crud import pwd_context
from schemas import schemas

from api.user_router import get_current_user
import boto3
from botocore.config import Config

settings = get_settings()

templates = Jinja2Templates(directory="templates")
router = APIRouter(
    prefix="/album",
)

boto_config = Config(
    signature_version = 'v4',
)
s3 = boto3.client("s3",
                  config=boto_config,
                  region_name='ap-northeast-2',
                  aws_access_key_id=settings.AWS_ACCESS_KEY,
                  aws_secret_access_key=settings.AWS_SECRET_KEY)


@router.get("")
async def upload_get(request: Request,
                     db: Session = Depends(get_db)):
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/user/login')
    
    email = token['email']
    
    user = crud.get_user_by_email(db=db, email=email)
    album_list = crud.get_uploads(db=db, user_id=user.user_id)
    
    return templates.TemplateResponse("album.html", {'request': request, 'token': token, 'album_list':album_list})


@router.get("/details")
async def upload_get_one(request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db)
    ):
    
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    video = crud.get_video(db=db, upload_id=upload_id)
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)
    frames = crud.get_frames(db=db, video_id=video.video_id)
    #obj = f"https://{settings.BUCKET}.s3.ap-northeast-2.amazonaws.com/{video_url}"
    obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': video.video_url},
                                    ExpiresIn=3600)
    obj_frame = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': '/'.join(video.video_url.split('/')[:-1])},
                                    ExpiresIn=3600)
    print(obj)
    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": uploaded.date,
        "upload_name": uploaded.name,
        "video_url": obj,
        "frames": frames
    }
    
    return templates.TemplateResponse("video.html", {'request': request, 'token': token, 'video_info': video_info})