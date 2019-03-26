# coding=utf-8

from flask import Flask, request

import character
import utils
from captcha_ocr import captcha_ocr

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


@app.route('/parse', methods=['GET', 'POST'])
def parse_captcha():
    text = 'unsupported'
    if request.method == 'POST':
        img = request.files['data']
        mark = request.args['mark']
        print('mark:%s' % mark)
        if mark == 'www.sc.10086.cn':
            model = utils.sc_10086_model
            with utils.graph.as_default():
                text = captcha_ocr.predict(model, img, 130, 50, character.ALPHANUMERIC_LOWER)
        elif mark == 'www.yn.10086.cn':
            model = utils.yn_10086_model
            with utils.graph.as_default():
                text = captcha_ocr.predict(model, img, 52, 20, character.NUMBER)
        elif mark == 'service.cq.10086.cn':
            model = utils.cq_10086_model
            with utils.graph.as_default():
                text = captcha_ocr.predict(model, img, 51, 22, character.ALPHANUMERIC_LOWER)
        return text
    return 'unsupported'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/hello')
def hello():
    return "hello world"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)
