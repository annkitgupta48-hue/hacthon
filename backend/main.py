from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(title="GestureSpeak AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def home():
    return {"message": "GestureSpeak AI backend is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}
