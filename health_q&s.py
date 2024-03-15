from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from ocr import getText


class HealthModel:
    def __init__(self, person_info=None):       #person_info为图片
        # 初始化模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained("./HuatuoGPT2-7B", device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("./HuatuoGPT2-7B", use_fast=True, trust_remote_code=True)
        # 初始化一个空的上下文列表
        self.messages = []
        self.person_info = self.get_Health_Info(person_info) if person_info != None else None
    
    
    def health_chat(self, question):
        if self.person_info != None:
            self.messages.append({"role": "user", "content": self.person_info})
            response = self.model.HuatuoChat(self.tokenizer, self.messages)
            return response
        else:
            self.messages.append({"role": "user", "content": question})
            response = self.model.HuatuoChat(self.tokenizer, self.messages)
            return response
        
    # 扫描体检报告
    def get_Health_Info(self, image):
        content = getText(image)
        content.append("以上是我的体检报告，请根据这份报告，分析我的健康状况，并给出建议")
        return ' '.join(content)
    
    # 清空聊天记录，开启新聊天
    def set_new_chat(self,):
        self.messages = []
    
    
    # 清空信息
    def delete_info(self,):
        self.person_info = None
        
