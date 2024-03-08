from datetime import timedelta, datetime

from fastapi import APIRouter, Response, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from utils.config import get_settings
from database.database import get_db, db_engine
from database import models
from database import crud
from database.crud import pwd_context
from schemas import schemas

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


router = APIRouter(prefix="/user")
templates = Jinja2Templates(directory="templates")

@router.get("/signup")
async def signup_get(request: Request):
    err_msg = {"user": None, "pw": None, "check_pw": None}
    return templates.TemplateResponse("signup.html", {"request": request, "err": err_msg})

@router.post("/signup")
async def signup_post(request: Request):
    body = await request.form()
    user, pw, check_pw = body["email"], body["pw"], body["check_pw"]
    err_msg = {"user": None, "pw": None, "check_pw": None}

    if not user:
        err_msg["user"] = "empty email"
    elif not pw:
        err_msg["pw"] = "empty password"
    elif pw != check_pw:
        err_msg["check_pw"] = "not equal password and check_password"
    else:
        session = Session(db_engine)
        user = session.query(models.User).filter(models.User.email == body['email']).first()
            
        if user:
            err_msg["user"] = "invalid email"
        else:
            user_info = models.User(email = body['email'],
                                    password = pwd_context.hash(body['pw']))
                
            session.add(user_info)
            session.commit()
            session.refresh(user_info)
            session.close()
            return RedirectResponse(url="/user/login")
    
    return templates.TemplateResponse("signup.html", {"request": request, "err": err_msg})

@router.get("/login")
async def login_get(request: Request):
    err_msg = {"user": None, "pw": None}
    return templates.TemplateResponse("login.html", {"request": request, "err": err_msg})

@router.post("/login")
async def login_post(request: Request):
    body = await request.form() 
    user, pw= body["email"], body["pw"]
    err_msg = {"user": None, "pw": None}

    if body.get("check_pw", None):
        return templates.TemplateResponse("login.html", {"request": request, "err": err_msg})

    if not user:
        err_msg["user"] = "empty email"
    elif not pw:
        err_msg["pw"] = "empty password"
    else:
        session = Session(db_engine)
        user_info_query = session.query(models.User).filter(models.User.email == body['email']).first()
        session.close()
        if not user_info_query:
            err_msg["user"] = "invalid email"
        elif not pwd_context.verify(body['pw'], user_info_query.password):
            err_msg["pw"] = "invalid password"
        else:
            return RedirectResponse(url="/")

    return templates.TemplateResponse("login.html", {"request": request, "err": err_msg})

@router.get("/logout")
async def logout_get(request: Request):
    access_token = request.cookies.get("access_token", None)    
    template = RedirectResponse(url="/")
    if access_token:
        template.delete_cookie(key="access_token")
    return template

# def get_current_user(token: str = Depends(oauth2_scheme),
#                      db: Session = Depends(get_db)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
#     else:
#         user = crud.get_user_by_email(db, email=email)
#         if user is None:
#             raise credentials_exception
#         return user


def get_current_user(request:Request):
	token = request.cookies.get("access_token", None)
	if token:
		payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
		user = payload.get("sub", None)
		return user
	else:
		return None