from typing import Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import ast

from datetime import timedelta, datetime
from sqlalchemy.orm import Session
from starlette import status


from api import user_router
from api import upload_router
from api import video_router
from api import real_time_router
from api import album_router

from db.database import get_db, db_engine
from models import models
from crud.crud import pwd_context
from core.config import get_settings
from jose import jwt, JWTError

templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

settings = get_settings()

@app.get("/")
async def main_get(request:Request):
	token = request.cookies.get("access_token", None)

	if token:
		payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
		user = payload.get("sub", None)
		return templates.TemplateResponse("main.html", {'request': request, 'token': user})
	else:
		return templates.TemplateResponse("main.html", {'request': request, 'token': token})

@app.post("/")
async def main_post(request: Request):
	body = await request.form()
	user = body["email"]
	# user_info_query = Session(db_engine).query(models.User).filter(models.User.email == body['email']).first()
	data = {
        "sub": user,
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
	token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
	
	template_response = templates.TemplateResponse('main.html', {'request': request, 'token': user})
 
    # 쿠키 저장
	template_response.set_cookie(key="access_token", value=token, expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES), httponly=True)
    
	return template_response

app.include_router(user_router.router)
app.include_router(upload_router.router)
app.include_router(album_router.router)
app.include_router(video_router.router)
app.include_router(real_time_router.router)

if __name__ == '__main__':
	uvicorn.run("main:app", host='0.0.0.0', port=30081, reload=True)