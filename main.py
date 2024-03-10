import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('./MiniCPM-V', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V', trust_remote_code=True)
model.eval()


# 输入图片询问或者直接询问
def main(img=None):
    if img:
        image = Image.open(img).convert('RGB')
    else:
        image = Image.new('RGB', (0, 0))
    context = []
    while 1:
        question = input("请输入问题：")
        if question == 'q':
            return
        msgs = [{'role': 'user', 'content': question}]
        res, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=context,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )
        return res




