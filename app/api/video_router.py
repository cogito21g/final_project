from datetime import timedelta, datetime, date
from typing import Optional
import ast

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Cookie
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

settings = get_settings()

templates = Jinja2Templates(directory="templates")
router = APIRouter(
    prefix="/video",
)

s3 = boto3.client("s3",aws_access_key_id=settings.AWS_ACCESS_KEY ,aws_secret_access_key=settings.AWS_SECRET_KEY)

@router.get("")
async def upload_get(request: Request):
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/user/login')

    return templates.TemplateResponse("video.html", {'request': request, 'token': token})

# @router.post("")
# async def upload_post(request: Request,
#                     name: str = Form(...), upload_file: UploadFile = File(...),
#                     date: date = Form(...),
#                     thr: float = Form(...),
#                     db: Session = Depends(get_db)):
#     token = request.cookies.get("access_token", None)
#     token = ast.literal_eval(token)
#     email = token['email']
    
#     user = crud.get_user_by_email(db=db, email=email)
#     _upload_create = schemas.UploadCreate(name=name, date=date,
#                                           user_id=user.user_id)
#     crud.create_upload(db=db, upload=_upload_create)
#     uploaded = crud.get_upload_id(db=db, user_id=user.user_id, name=name, date=date)
#     video_url = f"video/{user.user_id}/{uploaded.upload_id}/{upload_file.filename}"
    
#     _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
#     crud.create_video(db=db, video=_video_create)
    
#     s3_upload_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="video 를 s3 저장소 업로드에 실패했습니다."
#     )
#     try:
#         s3.upload_fileobj(
#             upload_file.file,
#             settings.BUCKET,
#             video_url
#         )
#     except:
#         raise s3_upload_exception
    
#     body = {
#         "user_id": user.user_id,
#         "email": user.email,
#         "upload_id": uploaded.upload_id,
#         "name": name,
#         "date": date,
#         "video_name": upload_file.filename,
#         "video_url": video_url
#     }
    
#     return templates.TemplateResponse("video.html", {'request': request, 'token': token, 'body': body})