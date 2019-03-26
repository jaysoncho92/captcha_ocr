# coding=utf-8
"""
直接使用三通道的图片训练,不转为灰度图
"""
import itertools
import os
import re
import random
import string
from collections import Counter
from os.path import join
import yaml
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model, load_model

f = open('./config/config_demo.yaml', 'r', encoding='utf-8')
cfg = f.read()
cfg_dict = yaml.load(cfg)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = cfg_dict['System']['GpuMemoryFraction']
session = tf.Session(config=config)
K.set_session(session)

# System config
TRAIN_SET_PTAH = cfg_dict['System']['TrainSetPath']
VALID_SET_PATH = cfg_dict['System']['TestSetPath']
TEST_SET_PATH = cfg_dict['System']['TestSetPath']
MAX_TEXT_LEN = cfg_dict['System']['MaxTextLenth']
IMG_W = cfg_dict['System']['IMG_W']
IMG_H = cfg_dict['System']['IMG_H']
MODEL_NAME = cfg_dict['System']['ModelName']
LABEL_REGEX = cfg_dict['System']['LabelRegex']
ALPHABET = cfg_dict['System']['Alphabet']

# NeuralNet config
RNN_SIZE = cfg_dict['NeuralNet']['RNNSize']
DROPOUT = cfg_dict['NeuralNet']['Dropout']

# TrainParam config
MONITOR = cfg_dict['TrainParam']['EarlyStoping']['monitor']
PATIENCE = cfg_dict['TrainParam']['EarlyStoping']['patience']
MODE = cfg_dict['TrainParam']['EarlyStoping']['mode']
BASELINE = cfg_dict['TrainParam']['EarlyStoping']['baseline']
EPOCHS = cfg_dict['TrainParam']['Epochs']
BATCH_SIZE = cfg_dict['TrainParam']['BatchSize']
TEST_BATCH_SIZE = cfg_dict['TrainParam']['TestBatchSize']
TEST_SET_NUM = cfg_dict['TrainParam']['TestSetNum']


def get_counter(dirpath):
    letters = ''
    lens = []
    for root, dirs, files in os.walk(dirpath):
        for filename in files:
            m = re.search(LABEL_REGEX, filename, re.M | re.I)
            description = m.group(1)
            lens.append(len(description))
            letters += description
    print('Max plate length in "%s":' % dirpath, max(Counter(lens).keys()))
    return Counter(letters)


c_val = get_counter(VALID_SET_PATH)
c_train = get_counter(TRAIN_SET_PTAH)
letters_train = set(c_train.keys())
letters_val = set(c_val.keys())
print('letters_train: %s' % ''.join(sorted(letters_train)))
print('letters_val: %s' % ''.join(sorted(letters_val)))
if letters_train == letters_val:
    print('Letters in train and val do match')
else:
    raise Exception('Letters in train and val don\'t match')
# print(len(letters_train), len(letters_val), len(letters_val | letters_train))
# letters = sorted(list(letters_train))
# letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
letters = ALPHABET
if len(letters) == 0:
    letters = string.digits + string.ascii_uppercase + string.ascii_lowercase
class_num = len(letters) + 1   # plus 1 for blank
print('Alphabet Letters:', ''.join(letters))


# Input data generator

def labels_to_text(labels):
    return ''.join([letters[int(x)] if int(x) != len(letters) else '' for x in labels])


def text_to_labels(text):
    return [letters.find(x) if letters.find(x) > -1 else len(letters) for x in text]


def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True


class TextImageGenerator:

    def __init__(self,
                 dirpath,
                 tag,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=MAX_TEXT_LEN):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = dirpath
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                m = re.search(LABEL_REGEX, filename, re.M | re.I)
                description = m.group(1)
                if len(description) < MAX_TEXT_LEN:
                    description = description + '_' * (MAX_TEXT_LEN - len(description))
                # if is_valid_str(description):
                #     self.samples.append([img_filepath, description])
                self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

        # build data:
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, 3))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # cv2默认是BGR模式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # cv2 BGR转RGB
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :, :] = img
            self.texts.append(text)

    @staticmethod
    def get_output_size():
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 3, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 3])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                # img = img.T
                img = np.transpose(img, (1, 0, 2))

                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                text = text.replace("_", "")  # important step
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


tiger = TextImageGenerator(VALID_SET_PATH, 'val', IMG_W, IMG_H, 8, 4)

for inp, out in tiger.next_batch():
    print('Text generator output (data which will be fed into the neutral network):')
    print('1) the_input (image)')
    if K.image_data_format() == 'channels_first':
        img = inp['the_input'][0, 0, :, :]
    else:
        img = inp['the_input'][0, :, :, 0]

    # plt.imshow(img.T, cmap='gray')
    # plt.show()
    print('2) the_labels (plate number): %s is encoded as %s' %
          (labels_to_text(inp['the_labels'][0]), list(map(int, inp['the_labels'][0]))))
    # print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
    #       (inp['input_length'][0], tiger.img_w))
    print('4) label_length (length of plate number): %d' % inp['label_length'][0])
    break


# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


downsample_factor = 4


def train(img_w=IMG_W, img_h=IMG_H, dropout=DROPOUT, batch_size=BATCH_SIZE, rnn_size=RNN_SIZE):
    # Input Parameters
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 3)

    global downsample_factor
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator(TRAIN_SET_PTAH, 'train', img_w, img_h, batch_size, downsample_factor)
    tiger_val = TextImageGenerator(VALID_SET_PATH, 'val', img_w, img_h, batch_size, downsample_factor)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = BatchNormalization()(inner)  # add BN
    inner = Activation(act)(inner)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = BatchNormalization()(inner)  # add BN
    inner = Activation(act)(inner)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=None, name='dense1')(inner)
    inner = BatchNormalization()(inner)  # add BN
    inner = Activation(act)(inner)
    if dropout:
        inner = Dropout(dropout)(inner)  # 防止过拟合

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    inner = concatenate([gru_2, gru_2b])

    if dropout:
        inner = Dropout(dropout)(inner)  # 防止过拟合

    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(inner)
    y_pred = Activation('softmax', name='softmax')(inner)
    base_model = Model(inputs=input_data, outputs=y_pred)
    base_model.summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

    earlystoping = EarlyStopping(monitor=MONITOR, patience=PATIENCE, verbose=1, mode=MODE, baseline=BASELINE)
    train_model_path = './tmp/train_' + MODEL_NAME
    checkpointer = ModelCheckpoint(filepath=train_model_path,
                                   verbose=1,
                                   save_best_only=True)

    if os.path.exists(train_model_path):
        model.load_weights(train_model_path)
        print('load model weights:%s' % train_model_path)

    evaluator = Evaluate(model)
    model.fit_generator(generator=tiger_train.next_batch(),
                        steps_per_epoch=tiger_train.n,
                        epochs=EPOCHS,
                        initial_epoch=2,
                        validation_data=tiger_val.next_batch(),
                        validation_steps=tiger_val.n,
                        callbacks=[checkpointer, earlystoping, evaluator])

    base_model.save('./model/' + MODEL_NAME)
    print('----train end----')


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


class Evaluate(Callback):
    def __init__(self, model):
        self.accs = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(self.model)
        self.accs.append(acc)


# Test on validation images
def evaluate(model):
    global downsample_factor
    tiger_test = TextImageGenerator(TEST_SET_PATH, 'test', IMG_W, IMG_H, TEST_BATCH_SIZE, downsample_factor)

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output
    predict_model = Model(inputs=net_inp, outputs=net_out)

    equalsIgnoreCaseNum = 0.00
    equalsNum = 0.00
    totalNum = 0.00
    for inp_value, _ in tiger_test.next_batch():
        batch_size = inp_value['the_input'].shape[0]
        X_data = inp_value['the_input']
        # net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
        net_out_value = predict_model.predict(X_data)
        pred_texts = decode_batch(net_out_value)
        labels = inp_value['the_labels']
        texts = []
        for label in labels:
            text = labels_to_text(label)
            texts.append(text)

        for i in range(batch_size):
            # print('Predict: %s ---> Label: %s' % (pred_texts[i], texts[i]))
            totalNum += 1
            if pred_texts[i] == texts[i]:
                equalsNum += 1
            if pred_texts[i].lower() == texts[i].lower():
                equalsIgnoreCaseNum += 1
            else:
                print('Predict: %s ---> Label: %s' % (pred_texts[i], texts[i]))
        if totalNum >= TEST_SET_NUM:
            break
    print('---Result---')
    print('Test num: %d, accuracy: %.5f, ignoreCase accuracy: %.5f' % (
    totalNum, equalsNum / totalNum, equalsIgnoreCaseNum / totalNum))
    return equalsIgnoreCaseNum / totalNum


if __name__ == '__main__':
    train()
    test = True
    if test:
        model_path = './tmp/train_' + MODEL_NAME
        model = load_model(model_path, compile=False)
        evaluate(model)
    print('----End----')
