import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from ocr import getText


# 加载模型
tokenizer = AutoTokenizer.from_pretrained("./HuatuoGPT2-7B", use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./HuatuoGPT2-7B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)


# 扫描体检报告
def getInfo(img):
    content = getText(img)
    content.append("以上是我的体检报告，请根据这份报告，分析我的健康状况，并给出建议")
    return ' '.join(content)

# 启动聊天
def health_chat(person_info = None):
    messages = []
    # 传入体检信息，先分析，否则跳过
    if person_info:
        messages.append({"role": "user", "content": person_info})
        response = model.HuatuoChat(tokenizer, messages)
        return response
    while 1:
        m = input("请输入你的问题:(q退出)")
        if m == 'q':
            return
        messages.append({"role": "user", "content": m})
        response = model.HuatuoChat(tokenizer, messages)
        return response

