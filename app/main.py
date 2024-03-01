from typing import Union
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from api.album import album_router 
from api.user import user_router
from api.video import video_router
from api.upload import upload_router


templates = Jinja2Templates(directory="templates")

app = FastAPI()

@app.get("/")
async def home(request:Request):
	return templates.TemplateResponse("main.html", {'request': request})

@app.get('/login')
async def login(request:Request):
	return templates.TemplateResponse('login.html',{'request': request})

@app.get('/signup')
async def signup(request:Request):
	return templates.TemplateResponse('signup.html',{'request': request})

app.include_router(user_router.router)

@app.get('/upload')
async def upload(request:Request):
	return templates.TemplateResponse('upload.html',{'request': request})

app.include_router(upload_router.router)

@app.get('/album')
async def album(request:Request):
	return templates.TemplateResponse('album.html',{'request': request})

app.include_router(album_router.router)

if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=30011)