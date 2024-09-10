from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
# Mount the static directory to serve static files
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    print('Request for index page received')
    with open("login.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)




if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8008)