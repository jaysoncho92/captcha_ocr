# coding=utf-8
import os
import re

import yaml

f = open('./config/config_demo.yaml', 'r', encoding='utf-8')
cfg = f.read()
cfg_dict = yaml.load(cfg)


TRAIN_SET_PTAH = cfg_dict['System']['TrainSetPath']
LABEL_REGEX = cfg_dict['System']['LabelRegex']
pattern = re.compile(LABEL_REGEX)

ch = ['0', '1', '2', '7', 'i', 'j', 'l', 'o', 'q', 'z']

for root, dirs, files in os.walk(TRAIN_SET_PTAH):
    for filename in files:
        m = re.search(LABEL_REGEX, filename, re.M|re.I)
        label = m.group(1)
        # label = filename.split('_')[-1].split('.')[0]
        # if len(label) < 5:
        for i in ch:
            if i in label:
                print('filename:%s, label:%s' % (filename, label))
                # break
