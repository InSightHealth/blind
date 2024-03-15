from fastapi import  FastAPI, UploadFile
from pydantic import BaseModel
from ocr import getText
from health_model import  HealthModel
from main import MainModel
from speech_to_text import Speech_To_Text


app = FastAPI()

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
        speech_to_text_model = Speech_To_Text()
    text = speech_to_text_model.speech_to_text(mp3.file)
    return {'text': text}


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
