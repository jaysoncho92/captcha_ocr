# coding=utf-8
import itertools

import cv2
import numpy as np
from PIL import Image
from keras import backend as K

import utils


def decode_batch(out, character):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(character):
                outstr += character[c]
        ret.append(outstr)
        print('character: %s, out_index: %s, predict: %s' % (character, out_best, outstr))
    return ret


def predict(model, img, img_w, img_h, character):
    img = Image.open(img)
    if img.mode != 'RGB':
        print('convert img mode from [%s] to RGB' % img.mode)
        img = img.convert('RGB')  # 云南移动图片为CMYK模式, 需要转为RGB
    img_data = np.asarray(img)
    img_data = cv2.resize(img_data, (img_w, img_h))
    X_data = get_x_data(img_data, img_w, img_h)
    with utils.graph.as_default():
        net_out_value = model.predict(X_data)
    y_pred = decode_batch(net_out_value, character)
    return y_pred[0]


def get_x_data(img_data, img_w, img_h):
    img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32)
    img /= 255
    batch_size = 1
    if K.image_data_format() == 'channels_first':
        X_data = np.ones([batch_size, 1, img_w, img_h])
    else:
        X_data = np.ones([batch_size, img_w, img_h, 1])
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    X_data[0] = img
    return X_data


if __name__ == '__main__':
    # img = Image.open('./image/o7pi.jpg')
    # image_data = np.asarray(img)
    # ret = predict(utils.model_sc10086)
    # print(ret)
    print('captcha ocr main')
