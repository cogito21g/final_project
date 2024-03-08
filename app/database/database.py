from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from utils.config import get_settings

settings = get_settings()

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(
    settings.MYSQL_SERVER_USER,
    settings.MYSQL_SERVER_PASSWORD,
    settings.MYSQL_SERVER_IP,
    settings.MYSQL_SERVER_PORT,
    settings.MYSQL_DATABASE
)

SQLALCHEMY_DATABASE_URL_ASYNC = "mysql+aiomysql://{}:{}@{}:{}/{}".format(
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
        
# async_engine = create_async_engine(SQLALCHEMY_DATABASE_URL_ASYNC)

# async def get_async_db():
#     db = AsyncSession(bind=async_engine)
#     try:
#         yield db
#     finally:
#         await db.close()