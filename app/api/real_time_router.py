from datetime import timedelta, datetime, date
import pytz
from typing import Optional, List, Tuple
import ast
import os
import uuid
import base64
import asyncio
import cv2
import numpy as np
import json

from fastapi import APIRouter, Response, Request,\
    HTTPException, Form, UploadFile, File, Cookie, Query, WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosed
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends, BackgroundTasks
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from core.config import get_settings
from db.database import get_db, db_engine
from models import models
from models.rt_anomaly_detector import RT_AnomalyDetector
from crud import crud
from crud.crud import pwd_context
from schemas import schemas

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


detector = None

@router.get("")
async def realtime_get(request: Request):
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
    else:
        return RedirectResponse(url='/user/login')

    return templates.TemplateResponse("real_time.html", {'request': request, 'token': token})

@router.post("")
async def realtime_post(request: Request,
                    name: str = Form(...),
                    real_time_video: str = Form(...),
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
    _upload_create = schemas.UploadCreate(name=name, date=datetime, is_realtime=True, thr=thr, user_id=user.user_id)
    crud.create_upload(db=db, upload=_upload_create)
    
    # 지금 업로드된 id 획득, 클라이언트로부터 작성된 실시간 스트리밍 영상 url 획득
    uploaded = crud.get_upload_id(db=db, user_id=user.user_id, name=name, date=datetime)[-1]
    
    # db 에는 실시간임을 알 수 있게만 함
    video_url = f"{real_time_video}"
    _video_create = schemas.VideoCreate(video_url=video_url, upload_id=uploaded.upload_id)
    crud.create_video(db=db, video=_video_create)
    _complete_create = schemas.Complete(completed=True, upload_id=uploaded.upload_id)
    crud.create_complete(db=db, complete=_complete_create)
    
    # model inference 에서 사용할 정보
    info = {
        "user_id": user.user_id,
        "email": user.email,
        "upload_id": uploaded.upload_id,
        "name": uploaded.name,
        "date": uploaded.date,
        "threshold": uploaded.thr,
        "video_url": video_url,
        "video_id": crud.get_video(db=db, upload_id=uploaded.upload_id).video_id
    }
    
    redirect_url = f"/real_time/stream?user_id={info['user_id']}&upload_id={info['upload_id']}"

    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()

    try:
        video_info_str = await websocket.receive_text()
        print("Received video info:", video_info_str)
        video_info = json.loads(video_info_str)
        global detector
        if detector is None:
            detector = RT_AnomalyDetector(video_info, s3, settings, db, websocket)
            detector.ready()
        
        while True:
            timestamp = datetime.now(pytz.timezone('Asia/Seoul'))
            # Receive bytes from the websocket
            bytes = await websocket.receive_bytes()
            await detector.run(bytes, timestamp)
            
    except WebSocketDisconnect:
        await websocket.close()
    
    finally:
        try:
            detector.upload_score_graph_s3()
        except:
            pass
        detector = None

# db 에서 실시간에서 저장되는 frame url 불러오는 코드    
def fetch_data(db, upload_id):
    
    video = crud.get_video(db=db, upload_id=upload_id)
    frames = crud.get_frames_with_highest_score(db=db, video_id=video.video_id)
    frame_ids = [frame.frame_id for frame in frames]
    frame_urls = [frame.frame_url for frame in frames]
    frame_timestamps = [frame.time_stamp for frame in frames]
    frame_objs = []
    
    for frame_id, frame_url, frame_timestamp in zip(frame_ids, frame_urls, frame_timestamps):
        frame_obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': frame_url},
                                    ExpiresIn=3600)
        frame_objs.append((frame_id, frame_obj, frame_timestamp.strftime('%H:%M:%S')))

    return {"frame_urls": frame_objs}

@router.get("/fetch_data")
async def fetch_frame_data(upload_id: int = Query(...),
                           db: Session = Depends(get_db)):
    
    frame_data = fetch_data(db, upload_id)
    return frame_data


@router.get("/stream")
async def get_stream(request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db)
    ):
    
    token = request.cookies.get("access_token", None)
    if token:
        token = ast.literal_eval(token)
        
    video = crud.get_video(db=db, upload_id=upload_id)
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)
    
    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": uploaded.date.strftime('%Y-%m-%d %H:%M:%S'),
        "upload_name": uploaded.name,
        "thr": uploaded.thr,
        "video_id": video.video_id,
        "video_url": video.video_url
    }
    
    # video_info = json.dumps(video_info)
    
    return templates.TemplateResponse("stream.html", {'request': request, 'token': token, 'video_info': video_info})