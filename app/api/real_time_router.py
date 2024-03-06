from datetime import timedelta, datetime, date
import ast
import uuid
import os

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Query
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
    prefix="/real_time",
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
async def real_time_get(request: Request):
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/user/login')
    
    return templates.TemplateResponse("real_time.html", {'request': request, "token": token})

@router.post("")
async def upload_post(request: Request,
                    name: str = Form(...),
                    upload_file: UploadFile = File(...),
                    date: date = Form(...),
                    thr: float = Form(...),
                    db: Session = Depends(get_db)):
    token = request.cookies.get("access_token", None)
    token = ast.literal_eval(token)
    email = token['email']
    
    user = crud.get_user_by_email(db=db, email=email)
    # name 중복 예외처리 구현?
    # check_name = crud.get_upload_by_name(db=db, name=name)
    # if check_name:
    #     name = check_name.name + uuid.uuid4()
    _upload_create = schemas.UploadCreate(name=name, date=date,
                                          user_id=user.user_id)
    crud.create_upload(db=db, upload=_upload_create)
    uploaded = crud.get_upload_id(db=db, user_id=user.user_id, name=name, date=date)[-1]
    video_ext = os.path.splitext(upload_file.filename)[-1]
    video_name = uuid.uuid1()
    video_url = f"video/{user.user_id}/{uploaded.upload_id}/{video_name}{video_ext}"
    
    _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
    crud.create_video(db=db, video=_video_create)
    
    s3_upload_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="video 를 s3 저장소 업로드에 실패했습니다."
    )
    try:
        s3.upload_fileobj(
            upload_file.file,
            settings.BUCKET,
            video_url,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
    except:
        raise s3_upload_exception
    
    info = {
        "user_id": user.user_id,
        "email": user.email,
        "upload_id": uploaded.upload_id,
        "name": name,
        "date": date,
        "threshold": thr,
        "video_name": upload_file.filename,
        "video_url": video_url
    }
    
    print(info)
    
    redirect_url = f"/upload/video?user_id={info['user_id']}&upload_id={info['upload_id']}&name={info['name']}&date={info['date']}&video_name={info['video_name']}"

    return RedirectResponse(url=redirect_url)


@router.post("/video")
async def video_get(request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    name: str = Query(...),
    date: str = Query(...),
    video_name: str = Query(...),
    db: Session = Depends(get_db)
    ):
    
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    
    video_url = crud.get_video(db=db, upload_id=upload_id).video_url
    #obj = f"https://{settings.BUCKET}.s3.ap-northeast-2.amazonaws.com/{video_url}"
    obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': video_url},
                                    ExpiresIn=3600)
    print(obj)
    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "name": name,
        "date": date,
        "video_name": video_name,
        "video_url": obj
    }
    
    return templates.TemplateResponse("video.html", {'request': request, 'token': token, 'video_info': video_info})