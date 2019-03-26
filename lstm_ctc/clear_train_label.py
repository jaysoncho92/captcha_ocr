# coding=utf-8
import itertools
import os
import re

import cv2
import numpy as np
import yaml
from PIL import Image
from keras import backend as K

import model_utils

f = open('./config/config_demo.yaml', 'r', encoding='utf-8')
cfg = f.read()
cfg_dict = yaml.load(cfg)

ALPHABET = cfg_dict['System']['Alphabet']
TRAIN_SET_PTAH = cfg_dict['System']['TrainSetPath']
LABEL_REGEX = cfg_dict['System']['LabelRegex']
IMG_W = cfg_dict['System']['IMG_W']
IMG_H = cfg_dict['System']['IMG_H']
pattern = re.compile(LABEL_REGEX)


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
        # print('character: %s, out_index: %s, predict: %s' % (character, out_best, outstr))
    return ret


def predict(model, img, img_w, img_h, character):
    img = Image.open(img)
    img_data = np.asarray(img)
    img_data = cv2.resize(img_data, (img_w, img_h))
    X_data = get_x_data(img_data, img_w, img_h)
    with model_utils.graph.as_default():
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
    i = 0
    for root, dirs, files in os.walk(TRAIN_SET_PTAH):
        for filename in files:
            m = re.search(LABEL_REGEX, filename, re.M | re.I)
            label = m.group(1)
            img = os.path.join(root, filename)
            predict_label = predict(model_utils.hb_ac_10086, img, img_w=IMG_W, img_h=IMG_H, character=ALPHABET)
            i += 1
            if label != predict_label:
                new_filename = filename.replace(label, predict_label)
                dst = os.path.join(root, new_filename)
                # os.rename(img, dst)
                print('rename file: old_file: %s, new_file: %s' % (filename, new_filename))
                # break
    print('total num:', i)
