import tensorflow as tf
from keras.models import load_model

graph = tf.get_default_graph()

# load model
base_model = load_model('./model/base_model.model_weights.h5', compile=False)
# sc_10086_model = load_model('./model/sc_10086_v1.model_weights.hdf5', compile=False)
service_cq_10086_model = load_model('./model/service.cq.10086.cn.model_weights.h5', compile=False)
hb_ac_10086 = load_model('./model/hb.ac.10086.cn.model_weights.h5', compile=False)
