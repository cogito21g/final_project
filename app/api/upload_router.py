from datetime import timedelta, datetime, date
from typing import Optional
import ast
import os
import uuid
import asyncio

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Cookie, Query
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from core.config import get_settings
from db.database import get_db, db_engine
from inference import models
from inference.anomaly_detector import AnomalyDetector
from crud import crud
from crud.crud import pwd_context
from schemas import schemas

from api.user_router import get_current_user
import boto3
from botocore.config import Config

settings = get_settings()
print(settings)

templates = Jinja2Templates(directory="templates")
router = APIRouter(
    prefix="/upload",
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
async def upload_get(request: Request):
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/user/login')

    return templates.TemplateResponse("upload.html", {'request': request, 'token': token})

@router.post("")
async def upload_post(request: Request,
                    background_tasks: BackgroundTasks,
                    name: str = Form(...),
                    upload_file: UploadFile = File(...),
                    datetime: datetime = Form(...),
                    thr: float = Form(...),
                    db: Session = Depends(get_db)):
    
    # token 정보 가져오기 -> email 을 통해 user_id 획득
    token = request.cookies.get("access_token", None)
    token = ast.literal_eval(token)
    email = token['email']

    # email 을 통해 user 객체 획득(id, email 포함)
    user = crud.get_user_by_email(db=db, email=email)
    
    # Form 과 user_id 를 이용하여 upload row insert
    _upload_create = schemas.UploadCreate(name=name, date=datetime, is_realtime=False, thr=thr, user_id=user.user_id)
    crud.create_upload(db=db, upload=_upload_create)
    
    # 지금 업로드된 id 획득, 클라이언트로부터 업로드된 비디오 정보(이름, 확장자) 획득
    uploaded = crud.get_upload_id(db=db, user_id=user.user_id, name=name, date=datetime)[-1]
    video_ext = os.path.splitext(upload_file.filename)[-1]
    # s3 의 경우 비디오 이름이 같으면 중복 업로드가 되지 않으므로 uuid 활용
    video_name = uuid.uuid1()
    
    # model inference 에서 s3 에 올릴 주소 그대로 db 에 insert
    video_url = f"video/{user.user_id}/{uploaded.upload_id}/{video_name}{video_ext}"
    #print(video_url)
    _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
    crud.create_video(db=db, video=_video_create)
    _complete_create = schemas.Complete(completed=False, upload_id=uploaded.upload_id)
    crud.create_complete(db=db, complete=_complete_create)
    
    # model inference 에서 사용할 정보
    info = {
        "user_id": user.user_id,
        "email": user.email,
        "upload_id": uploaded.upload_id,
        "name": name,
        "date": datetime,
        "threshold": uploaded.thr,
        "video_name": upload_file.filename,
        "video_uuid_name": video_name,
        "video_ext": video_ext,
        "video_id": crud.get_video(db=db, upload_id=uploaded.upload_id).video_id
    }
    #print(info)
    
    background_tasks.add_task(run_model,
                              video_url, upload_file, info, s3, settings, db)  
    
    
    redirect_url = f"/album/details?user_id={info['user_id']}&upload_id={info['upload_id']}"

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

# background 에서 모델 실행
def run_model(video_url, upload_file, info, s3, settings, db):
    
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

    model = AnomalyDetector(video_file=video_url,
                            info=info,
                            s3_client=s3,
                            settings=settings,
                            db=db)
    model.run()
    
    crud.update_complete_status(db=db, upload_id=info['upload_id'])