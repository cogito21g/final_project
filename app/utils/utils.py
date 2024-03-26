from fastapi import HTTPException, status
import boto3
from botocore.config import Config

from utils.config import settings
from database import crud
from inference.anomaly_detector import AnomalyDetector

boto_config = Config(
    signature_version = 'v4',
)

s3 = boto3.client("s3",
                  config=boto_config,
                  region_name='ap-northeast-2',
                  aws_access_key_id=settings.AWS_ACCESS_KEY,
                  aws_secret_access_key=settings.AWS_SECRET_KEY)

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
    return 