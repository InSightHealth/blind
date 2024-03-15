import requests
import json
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

class CommonOcr(object):
    def __init__(self, img_path):
        self._app_id = '9b426bc3d4f5098c8cddda2baef5db6d'
        self._secret_code = 'e884c6d5163fb48b1d0601bc77e3a490'
        self._img_path = img_path

    def recognize(self):
        # 通用文字识别
        url = 'https://api.textin.com/ai/service/v2/recognize'
        head = {}
        try:
            image = get_file_content(self._img_path)
            head['x-ti-app-id'] = self._app_id
            head['x-ti-secret-code'] = self._secret_code
            result = requests.post(url, data=image, headers=head)
            return result.text
        except Exception as e:
            return e
        

def getText(img_url):   
    # 发送GET请求到图片URL
    response = requests.get(img_url)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用BytesIO将响应内容转换为字节对象
        image_bytes = BytesIO(response.content)
        
        # 创建一个临时文件
        with NamedTemporaryFile(delete=False) as tmp_file:
            # 将字节对象写入临时文件
            tmp_file.write(image_bytes.read())
            
            # 调用TextIn API来识别文字
            ans = CommonOcr(tmp_file.name).recognize()
            
            # 将识别的文字转换为字典
            data = json.loads(ans)
            
            # 从结果中提取文字
            text_list = [line["text"] for line in data["result"]["lines"]]
            
            # 返回识别到的文字列表
            return text_list
    else:
        # 如果请求失败，返回一个错误信息
        return f"获取图片失败，状态码：{response.status_code}"
