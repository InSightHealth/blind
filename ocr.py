import requests
import json


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


def getText(img):
    response = CommonOcr(img)
    ans = response.recognize()
    data = json.loads(ans)  # 将str对象转换为字典
    text_list = []  # 创建一个空列表用于存储text的内容
    for line in data["result"]["lines"]:  # 遍历字典中的lines列表
        text_list.append(line["text"])  # 将每个line中的text添加到列表中
    return text_list
