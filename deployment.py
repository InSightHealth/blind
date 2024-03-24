from io import BytesIO
from fastapi import  FastAPI, UploadFile
from pydantic import BaseModel
from ocr import getText
from health_model import  HealthModel
from main import MainModel
import numpy as np
import soundfile as sf
from transformers import pipeline
import librosa

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    image: str | None = None


class Img(BaseModel):
    image: str
    
    
class ClearStatus(BaseModel):
    main_model: int  = 0   # 0不清除记录
    health_model: int = 0


main_model = None
health_model = None
speech_to_text_model = None


# 主要模型，出行，日常
@app.post("/chatbot")
async def chatbot(question: GenerateRequest):
    global main_model
    if main_model == None:
            main_model = MainModel()
    if question.image == None:
        response = main_model.question_and_answer(question.prompt)
    else:
        response = main_model.question_and_answer(question.prompt, question.image)
    return {"response": response}


# 健康问答
@app.post("/healthbot")
async def health_chat_bot(question: GenerateRequest):
    global health_model
    if health_model == None and question.image != None:
        health_model = HealthModel(question.image)
    elif health_model == None and question.image == None:
        health_model = HealthModel()
    response = health_model.health_chat(question.prompt)
    return {"response": response}



# 语音转文字
@app.post("/speechtotext")
async def speech_to_text(mp3: UploadFile):
    global speech_to_text_model
    if speech_to_text_model == None:
        speech_to_text_model = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    # 读取MP3文件为二进制数据
    mp3_data = await mp3.read()
    # 将二进制数据转换为numpy数组
    audio, _ = librosa.load(BytesIO(mp3_data), mono=True)    
    return speech_to_text_model(audio)


# 清空聊天记录
@app.post('/clear')
async def clear(status: ClearStatus):
    if status.main_model != 0:
        if main_model != None:
            main_model.set_new_chat()
    if status.health_model != 0:
        if health_model != None:
            health_model.set_new_chat()


@app.post('/ocr')
async def ocr(image: Img):
    return getText(image.image)


