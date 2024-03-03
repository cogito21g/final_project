from datetime import timedelta, datetime

from fastapi import APIRouter, Request, HTTPException
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

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

signup_router = APIRouter(prefix='/login')
templates = Jinja2Templates(directory="templates")


@signup_router.post('')
async def user_create(request: Request):
    body = await request.form()     # html에서 받은 form 데이터


    with Session(db_engine) as session:
        # User 테이블에 입력한 이메일이 있는지 확인 -> 있다면 회원가입 화면 다시 띄우기
        user = session.query(models.User).filter(models.User.email == body['email']).first()
        if user:
            return RedirectResponse(url='/signup')
        
        # 입력 받은 비밀번호 2개가 같은지 확인 -> 다르다면 회원가입 화면 다시 띄우기
        if body['pw'] != body['check_pw']:
            return RedirectResponse(url='/signup')
        
        # 입력받은 값들을 User 테이블에 저장
        user_info = models.User(email = body['email'],
                                password = pwd_context.hash(body['pw']))
        
        session.add(user_info)
        session.commit()
        session.refresh(user_info)
    
    return templates.TemplateResponse("login.html", {"request": request, "body": body})


# @router.post("/", response_model=schemas.Token)
# def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
#                            db: Session = Depends(get_db)):

#     # check user and password
#     user = crud.get_user_by_email(db, form_data.username)
#     if not user or not pwd_context.verify(form_data.password, user.password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="이메일 혹은 비밀번호가 올바르지 않습니다.",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     # make access token
#     data = {
#         "sub": user.email,
#         "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     }
#     access_token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "email": user.email
#     }


def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    else:
        user = crud.get_user_by_email(db, email=email)
        if user is None:
            raise credentials_exception
        return user