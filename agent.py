from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from agent_logic import ask_agent
import json
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

chat_history = []
DATA_DIR = "data"
CHAT_LOG_PATH = os.path.join(DATA_DIR, "chat_log.json")

if os.path.exists(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "r") as f:
        chat_history = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def chat_get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})

@app.post("/", response_class=HTMLResponse)
async def chat_post(request: Request, user_input: str = Form(...)):
    reply = ask_agent(user_input, chat_history)
    chat_history.append({"user": user_input, "bot": reply})
    save_chat_log()
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})

def save_chat_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHAT_LOG_PATH, "w") as f:
        json.dump(chat_history, f, indent=2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=8000, reload=True)
