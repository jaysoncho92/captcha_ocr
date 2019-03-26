import tensorflow as tf
from keras.models import load_model

graph = tf.get_default_graph()

# load model
base_model = load_model('./captcha_ocr/model/base_model.model_weights.h5', compile=False)
sc_10086_model = load_model('./captcha_ocr/model/sc_10086_v1.model_weights.hdf5', compile=False)
yn_10086_model = load_model('./captcha_ocr/model/www.yn.10086.cn.model_weights.h5', compile=False)
cq_10086_model = load_model('./captcha_ocr/model/service.cq.10086.cn.model_weights.h5', compile=False)
