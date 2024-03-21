from pydantic_settings import BaseSettings
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import boto3
from botocore.config import Config

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "55c84cbfa7f9e183da2179cb34cc45526bea05ee80b5bef66ed950534730bf5d"
    ALGORITHM: str = "HS256"
    # 60 minutes * 24 hours * 7 days = 7 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    
    MYSQL_SERVER_IP : str
    MYSQL_SERVER_PORT : int
    MYSQL_SERVER_USER : str
    MYSQL_SERVER_PASSWORD : str
    MYSQL_DATABASE : str
    
    AWS_ACCESS_KEY : str
    AWS_SECRET_KEY : str
    BUCKET : str = "cv06-bucket2"

    SMTP_ADDRESS : str
    SMTP_PORT : int
    MAIL_ACCOUNT : str
    MAIL_PASSWORD : str
    
    class Config:
        env_file = ".env"
        
@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(
    settings.MYSQL_SERVER_USER,
    settings.MYSQL_SERVER_PASSWORD,
    settings.MYSQL_SERVER_IP,
    settings.MYSQL_SERVER_PORT,
    settings.MYSQL_DATABASE
)

db_engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


boto_config = Config(
    signature_version = 'v4',
)
s3 = boto3.client("s3",
                  config=boto_config,
                  region_name='ap-northeast-2',
                  aws_access_key_id=settings.AWS_ACCESS_KEY,
                  aws_secret_access_key=settings.AWS_SECRET_KEY)