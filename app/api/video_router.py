from datetime import timedelta, datetime, date
from typing import Optional

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Cookie
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status

from utils.config import get_settings
from database.database import get_db, db_engine
from database import crud
from database.crud import pwd_context
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
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url='/user/login')

    return templates.TemplateResponse("video.html", {'request': request, 'token': user})

# @router.post("")
# async def upload_post(request: Request,
#                     name: str = Form(...), upload_file: UploadFile = File(...),
#                     date: date = Form(...),
#                     thr: float = Form(...),
#                     db: Session = Depends(get_db)):
    #   email = get_current_user(request)
    
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
    
#     return templates.TemplateResponse("video.html", {'request': request, 'token': email, 'body': body})