from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from db.database import get_db, get_async_db
from crud import crud
from schemas import schemas
from models.models import User

router = APIRouter(
    prefix="/album",
)


@router.get("/", response_model=schemas.Upload)
def album_list(db: Session = Depends(get_db),
                  page: int = 0, size: int = 10):
    total, _upload_list = crud.get_uploads(
        db, skip=page * size, limit=size)
    return {
        'total': total,
        'upload_list': _upload_list
    }
