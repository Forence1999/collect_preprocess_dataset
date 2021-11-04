import os
import argparse
import six
from lib.confusion_matrix_pretty_print import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.python.keras.losses import Loss
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from lib import utils, models_tf
from lib.mi_data import load_hole_dataset, one_hot_encoder
import shutil
from collections import Counter
from tensorflow.python.platform import tf_logging as logging
from calculate_doa import *

if __name__ == '__main__':
    
    # setting keras
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    K.set_image_data_format('channels_first')
    
    # model_configure & dataset_configure
    md_config = MODEL_CONFIG(name='EEGNet', batch_size=64, epochs=100, dropoutRate=0.25, earlystop_patience=50,
                             TL_method='no_EA', CSP=False, filters=32, normalization='sample-wise', )
    ds_config = DATASET_CONFIG(ds_dir='../dataset/hole/', seg_len='256ms', sample_rate=16000,
                               preproccess_type='normalized_denoise_nsnet2', gcc_phat=True, nb_class=8, )
    
    K.clear_session()
    # load data
    idx = list(range(ds_config.nb_sbj))
    x, y = load_hole_dataset(idx, ds_config.ds_path, shuffle=True,
                             normalization=md_config.normalization, one_hot=False)
    
    model = keras.models.load_model('G:\SmartWalker\SSL\code\model\EEGNet\ckpt')
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=-1)
    plot_confusion_matrix_from_data(y, y_pred)
    K.clear_session()
