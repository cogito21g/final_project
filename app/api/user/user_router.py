from datetime import timedelta, datetime

from fastapi import APIRouter, Response, Request, HTTPException
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
login_router = APIRouter(prefix='/main')
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


@login_router.post('')
async def login_and_access_token(request: Request):

    # login.thml form에서 받은 email, password가 User 테이블에 있는지 확인하기
    # 입력받은 email이 없거나, 비밀번호가 틀린 경우 로그인 화면을 다시 띄웁니다.
    body = await request.form()
    user_info_query = Session(db_engine).query(models.User).filter(models.User.email == body['email']).first()

    if not user_info_query or not pwd_context.verify(body['password'], user_info_query.password):
        return templates.TemplateResponse('login.html', {'request': request})
        # RedirectResponse로 화면을 다시 띄우려 했지만, POST login이 회원가입하면서 바로 넘어오는 로직으로 구현되어 있어 템플릿으로 구현했습니다.
    
    # make access token
    data = {
        "sub": user_info_query.email,
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    token = {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user_info_query.email
    }

    template_response = templates.TemplateResponse('main.html', {'request': request, 'token': token})
    # 쿠키 저장
    template_response.set_cookie(key="access_token", value=token, expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES), httponly=True)
    
    return template_response


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