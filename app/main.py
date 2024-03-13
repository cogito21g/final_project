from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from datetime import timedelta, datetime

from api import user_router, upload_router, real_time_router, album_router

from database.database import get_db, db_engine
from database import models
from database.crud import pwd_context
from utils.config import get_settings
from jose import jwt


templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.28.224.136:30011"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()


@app.get("/")
async def main_get(request:Request):

	user = user_router.get_current_user(request)
	
	return templates.TemplateResponse("main.html", {'request': request, 'token': user.email})

@app.post("/")
async def main_post(request: Request):
	body = await request.form()
	user = body["email"]
	data = {
        "sub": user,
        "exp": datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    }
	token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
	
	template_response = templates.TemplateResponse('main.html', {'request': request, 'token': user.email})
 
    # 쿠키 저장
	template_response.set_cookie(key="access_token", value=token, expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES), httponly=True)
    
	return template_response

app.include_router(user_router.router)
app.include_router(upload_router.router)
app.include_router(album_router.router)
app.include_router(real_time_router.router)

if __name__ == '__main__':
	uvicorn.run("main:app", host='0.0.0.0', port=30081, reload=True)