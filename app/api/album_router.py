from datetime import timedelta, datetime, date
from typing import Optional

from fastapi import APIRouter, Response, Request, HTTPException, Form, UploadFile, File, Cookie, Query, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status


from utils.config import get_settings
from database.database import get_db, db_engine

from database import crud, models
from database.crud import pwd_context

from api.user_router import get_current_user
import boto3
from botocore.config import Config
import json


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
async def upload_get(request: Request, db: Session = Depends(get_db)):
    email = get_current_user(request)
    if not email:
        return RedirectResponse(url='/user/login')
        
    user = crud.get_user_by_email(db=db, email=email)
    album_list = crud.get_uploads(db=db, user_id=user.user_id)
    
    return templates.TemplateResponse("album.html", {'request': request, 'token': email, 'album_list':album_list})


@router.post("")
async def modify_name(request: Request,
                      check_code: str = Form(...),
                      upload_id: Optional[int] = Form(...),
                      origin_name: Optional[str] =  Form(None),
                      new_name: Optional[str] = Form(None),
                      is_real_time: Optional[bool] = Form(None),
                      db: Session = Depends(get_db)):
    email = get_current_user(request)

    if check_code == "edit":
        upload_info = db.query(models.Upload).filter((models.Upload.name == origin_name) & 
                                    (models.Upload.upload_id == upload_id)).first()
        upload_info.name = new_name

        db.add(upload_info)
        db.commit()
        db.refresh(upload_info)
    elif check_code == "delete":
        # upload 테이블에서만 지우면 SQLAlchemy relationship cascade 설정에 의해
        # 자식 테이블의 관련된 데이터도 삭제가 된다.
        upload_info = crud.get_upload(db, upload_id)
        if upload_info:
            db.delete(upload_info)

        db.commit()
     

    # album_list를 만들고 끝.
    user = crud.get_user_by_email(db=db, email=email)
    album_list = crud.get_uploads(db=db, user_id=user.user_id)

    return templates.TemplateResponse("album.html", {'request': request, 'token': email, 'album_list': album_list})




@router.get("/details")
async def upload_get_one(request: Request,
    user_id: int = Query(...),
    upload_id: int = Query(...),
    db: Session = Depends(get_db)
    ):

    user = get_current_user(request)
        
    if not crud.get_complete(db=db, upload_id=upload_id).completed:
        return templates.TemplateResponse("video.html", {'request': request, 'token': user, 'video_info': {}, 'loading': True})
        
    video = crud.get_video(db=db, upload_id=upload_id)
    uploaded = crud.get_upload(db=db, upload_id=video.upload_id)
    #frames = crud.get_frames(db=db, video_id=video.video_id)
    frames = crud.get_frames_with_highest_score(db=db, video_id=video.video_id)
    frame_ids = [frame.frame_id for frame in frames]
    frame_urls = [frame.frame_url for frame in frames]
    frame_timestamps = [frame.time_stamp for frame in frames]
    frame_objs = []
    
    #obj = f"https://{settings.BUCKET}.s3.ap-northeast-2.amazonaws.com/{video_url}"
    video_obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': video.video_url},
                                    ExpiresIn=3600)
    
    for frame_id, frame_url, frame_timestamp in zip(frame_ids, frame_urls, frame_timestamps):
        frame_obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': frame_url},
                                    ExpiresIn=3600)
        frame_objs.append((frame_id, frame_obj, frame_timestamp.strftime('%H:%M:%S')))
    
    score_graph_url = '/'.join(frame_urls[0].split('/')[:-1]) + '/score_graph.png'
    #print(f'score_graph_url >>> {score_graph_url}')
    score_obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': score_graph_url},
                                    ExpiresIn=3600)
    
    #print(obj)
    
    video_info = {
        "user_id": user_id,
        "upload_id": upload_id,
        "date": uploaded.date.strftime('%Y-%m-%d %H:%M:%S'),
        "upload_name": uploaded.name,
        "video_id": video.video_id,
        "video_url": video_obj,
        "frame_urls": frame_objs,
        "score_url": score_obj
    }
    
    #video_info = json.dumps(video_info)
    #print(video_info.video_url)
    #print(frame_objs[0])
    
    return templates.TemplateResponse("video.html", {'request': request, 'token': token, 'video_info': video_info, 'loading': False})


@router.get("/details/images")
async def image_get(request: Request,
                    frame_id: int = Query(...),
                    db: Session = Depends(get_db)
                    ):
    
    user = get_current_user(request)
    frame = crud.get_frame(db=db, frame_id=frame_id)
    frame_obj = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': settings.BUCKET,
                                            'Key': frame.frame_url},
                                    ExpiresIn=3600)
    print(frame_obj)
    print(frame.box_kp_json)
    frame_info = {
        'frame_url': frame_obj,
        'time_stamp': frame.time_stamp,
        'frame_json': frame.box_kp_json
    }
    
    return templates.TemplateResponse("frame.html", {'request': request, 'token': user, 'frame_info': frame_info})
    