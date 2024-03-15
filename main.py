import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO


class MainModel:
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained('./MiniCPM-V', 
                                  trust_remote_code=True,
                                  device_map="cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V', trust_remote_code=True)
        self.context = []
        self.temperature = 0.6
        
    # 问答    
    def question_and_answer(self, question, image=None):
        if image != None:
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.new('RGB', (0, 0))
        msgs = [{'role': 'user', 'content': question}]
        res, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=self.context,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=self.temperature
        )
        self.context.append(context)
        return res
    
    # 清空聊天记录，开启新聊天
    def set_new_chat(self,):
        self.context = []

