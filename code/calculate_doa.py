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


def calculate_weighted_acc(x, y, model, class_weight):
    y_pred = model.predict(x)
    weighted_acc = utils.calculate_class_weighted_accuracy(y, y_pred, class_weight)
    
    return weighted_acc


class weighted_acc_callback(Callback):
    def __init__(self, train_data, val_data, test_data, model):
        super(weighted_acc_callback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model = model
        self.train_weighted_acc = []
        self.val_weighted_acc = []
        self.test_weighted_acc = []
        self.train_cls_weight = calculate_class_weight(train_data[1])
        self.val_cls_weight = calculate_class_weight(val_data[1])
        if test_data is not None:
            self.test_cls_weight = calculate_class_weight(test_data[1])
    
    def on_epoch_end(self, epoch, logs={}):
        self.train_weighted_acc.append(
            calculate_weighted_acc(self.train_data[0], self.train_data[1], self.model, self.train_cls_weight))
        self.val_weighted_acc.append(
            calculate_weighted_acc(self.val_data[0], self.val_data[1], self.model, self.val_cls_weight))
        if self.test_data is not None:
            self.test_weighted_acc.append(
                calculate_weighted_acc(self.test_data[0], self.test_data[1], self.model, self.test_cls_weight))
        else:
            self.test_weighted_acc.append(0.)
        print('Weighted Acc: \n train: {:.2f} val: {:.2f} test: {:.2f}'.format(
            self.train_weighted_acc[-1] * 100., self.val_weighted_acc[-1] * 100.,
            self.test_weighted_acc[-1] * 100. if self.test_data is not None else 0., ))


class customized_earlystopping(EarlyStopping):
    def __init__(self, x_val, y_val, model, monitor='val_acc', min_delta=0, patience=0, verbose=0,
                 mode='max', baseline=None, restore_best_weights=False):
        super(customized_earlystopping, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience,
                                                       verbose=verbose, mode=mode, baseline=baseline,
                                                       restore_best_weights=restore_best_weights)
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.val_cls_weight = calculate_class_weight(y_val)
    
    def get_monitor_value(self, logs):
        return calculate_weighted_acc(self.x_val, self.y_val, self.model, self.val_cls_weight)


class customized_modelcheckpoint(ModelCheckpoint):
    def __init__(self, x_val, y_val, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch',
                 options=None, ):
        super(customized_modelcheckpoint, self).__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                                                         save_best_only=save_best_only,
                                                         save_weights_only=save_weights_only, mode=mode,
                                                         save_freq=save_freq, options=options, )
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.val_cls_weight = calculate_class_weight(y_val)
    
    def get_monitor_value(self, ):
        return calculate_weighted_acc(self.x_val, self.y_val, self.model, self.val_cls_weight)
    
    def _save_model(self, epoch, logs):
        """Saves the model.
  
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)
            
            try:
                if self.save_best_only:
                    current = self.get_monitor_value()
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                
                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e


def build_model(model_name='EEGNet', nb_classes=2, Chans=64, Samples=240, dropoutRate=0.25):
    if model_name == 'FCN':
        return models_tf.EEGNet(nb_classes=nb_classes, Chans=Chans, SamplePoints=Samples, dropoutRate=dropoutRate)
    elif model_name == 'EEGNet':
        return models_tf.EEGNet(nb_classes=nb_classes, Chans=Chans, SamplePoints=Samples, dropoutRate=dropoutRate)
    elif model_name == 'DeepConvNet':
        return models_tf.DeepConvNet(nb_classes=nb_classes, Chans=Chans, SamplePoints=Samples, dropoutRate=dropoutRate)
    elif model_name == 'ShallowConvNet':
        return models_tf.ShallowConvNet(nb_classes=nb_classes, Chans=Chans, SamplePoints=Samples,
                                        dropoutRate=dropoutRate)
    else:
        raise Exception('No such model:{}'.format(model_name))


def calculate_class_weight(y):
    # return dictionary mapping class indices (integers) to a weight (float) value
    y = np.array(y, np.int)
    y_num = dict(Counter(y))
    
    for key in list(y_num.keys()):
        y_num[key] = 1. / (y_num[key] / len(y))
    total = np.sum(list(y_num.values()))
    for key in list(y_num.keys()):
        y_num[key] = y_num[key] / total
    
    return y_num


def train_model(train_dataset, val_dataset, test_dataset, model, model_path="./model", batch_size=32,
                epochs=300, patience=100):
    x_train, y_train = utils.shuffle_data(train_dataset)
    train_class_weight = calculate_class_weight(y_train)
    x_val, y_val = val_dataset
    if test_dataset is not None:
        x_test, y_test = test_dataset
    
    nb_train_batch = np.ceil(len(x_train) / batch_size)
    
    # Train Model
    print('Start to train model')
    lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01,
                                                    decay_steps=nb_train_batch * epochs)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=nb_train_batch, decay_rate=0.99,
    #                                                              staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc', ])
    ### TensorBoard
    # log_dir = "./log"
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # Tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    best_ckpt = customized_modelcheckpoint(x_val=x_val, y_val=y_val, model=model, filepath=model_path,
                                           monitor='val_acc', save_best_only=True, mode='max')
    early_stop = customized_earlystopping(x_val=x_val, y_val=y_val, model=model, monitor='val_acc', mode='max',
                                          patience=patience)
    test_callback = weighted_acc_callback(train_dataset, val_dataset, test_dataset, model=model)
    
    history = model.fit(x_train, y_train, class_weight=train_class_weight, epochs=epochs, batch_size=batch_size,
                        shuffle=True, callbacks=[best_ckpt, early_stop, test_callback], verbose=0, )
    
    # return {
    #     'train_acc' : np.array(history.history['acc'], dtype=np.float32),
    #     'train_loss': np.array(history.history['loss'], dtype=np.float32),
    #     'val_acc'   : np.array(history.history['val_acc'], dtype=np.float32),
    #     'val_loss'  : np.array(history.history['val_loss'], dtype=np.float32),
    # }
    return {
        'train_acc': np.array(test_callback.train_weighted_acc, dtype=np.float32),
        'val_acc'  : np.array(test_callback.val_weighted_acc, dtype=np.float32),
        'test_acc' : np.array(test_callback.test_weighted_acc, dtype=np.float32),
    }


def eva_model(model, dataset):
    '''
    cal_acc
    input:
    model_name,model, model_path,data_name,x_test,y_test

    output:
    acc
    y_pred
    '''
    x_test, y_test = dataset
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    y_test = np.squeeze(y_test)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
    
    return acc, y_pred, y_pred_prob


def compute_target_gradient(x, model, target):
    """
    Computes the gradient of the input image batch wrt the target output class.

    Note, this gradient is only ever computed from the <Student> model,
    and never from the <Teacher/Attacked> model when using this version.

    Args:
        x: batch of input of shape [B, T, C]
        model: classifier model
        target: integer id corresponding to the target class

    Returns:
        the output of the model and a list of gradients of shape [B, T, C]
    """
    with tf.GradientTape() as tape:
        tape.watch(x)  # need to watch the input tensor for grad wrt input
        out = model(x, training=False)  # in evaluation mode
        target_out = out[:, target]  # extract the target class outputs only
    
    image_grad = tape.gradient(target_out, x)  # compute the gradient
    
    return out, image_grad


class DATASET_CONFIG:
    def __init__(self, ds_dir='../dataset/hole/', seg_len='1s', sample_rate=16000,
                 preproccess_type='normalized_denoise_nsnet2', gcc_phat=True, nb_class=8, ):
        self.seg_len = seg_len
        self.sample_rate = sample_rate
        self.preproc_type = preproccess_type
        self.gcc_phat = gcc_phat
        self.nb_sbj = None
        self.nb_class = nb_class
        self.nb_channel = None
        self.nb_samplepoint = None
        self.ds_path = None
        
        self.ds_path = os.path.normpath(os.path.join(ds_dir,
                                                     self.seg_len + '_' + str(self.sample_rate) + '_' +
                                                     self.preproc_type + '_gcc_phat' * self.gcc_phat + '.npz', ))
        x = np.load(file=self.ds_path, allow_pickle=True)['x']
        self.nb_sbj = len(x)
        (_, self.nb_channel, self.nb_samplepoint) = x[0].shape
        del x


class MODEL_CONFIG:
    def __init__(self, md_dir='./model', name='EEGNet', batch_size=32, epochs=1000, dropoutRate=0.25,
                 earlystop_patience=50,
                 TL_method='no_EA',
                 CSP=False, filters=32, normalization=None, ):
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropoutRate = dropoutRate
        self.earlystop_patience = earlystop_patience
        self.TL_method = TL_method
        self.CSP = CSP
        self.filters = filters
        self.normalization = normalization
        self.md_dir = os.path.join(md_dir, self.name)
        self.ckpt_dir = os.path.join(self.md_dir, 'ckpt')
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)


class Gauss_sparse_categorical_crossentropy_Loss(Loss):
    def __init__(self, nb_class=8, mean=0, sigma=1, label_smoothing=0):
        super(Gauss_sparse_categorical_crossentropy_Loss, self).__init__()
        self.nb_class = nb_class
        weight = np.roll(
            np.linspace(-self.nb_class // 2, self.nb_class // 2, endpoint=self.nb_class % 2, num=self.nb_class),
            -self.nb_class // 2)
        weight = np.exp(- ((weight - mean) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
        weight = -weight / weight.max()
        self.weight = weight - weight.min()
        self.label_smoothing = label_smoothing
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        print('Using self_customized loss function')
        
        weight = np.roll(self.weight, y_true)
        onehot_labels = one_hot_encoder(y_true)
        if self.label_smoothing > 0:
            smooth_positives = 1.0 - self.label_smoothing
            smooth_negatives = self.label_smoothing / self.nb_class
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives
        
        cross_entropy = -tf.reduce_mean(weight * onehot_labels * tf.log(tf.clip.by_value(y_pred, 1e-10, 1.0)))
        
        return cross_entropy


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
    
    # set paths
    npz_path = os.path.join(md_config.md_dir, 'result.npz')
    img_path = os.path.join(md_config.md_dir, 'acc_loss.jpg')
    
    print("------------------train model------------------")
    K.clear_session()
    # load data ---- split datasets for train, val & test
    sbj_rand_idx = utils.get_shuffle_index(ds_config.nb_sbj)
    train_idx, val_idx, test_idx = sorted(sbj_rand_idx[0:7]), sorted(sbj_rand_idx[7:]), sorted(sbj_rand_idx[9:])
    print("-----{} for train----{} for val-----{} for test-----".format(train_idx, val_idx, test_idx))
    x_train, y_train = load_hole_dataset(train_idx, ds_config.ds_path, shuffle=True, split=None,
                                         normalization=md_config.normalization, one_hot=False)
    x_val, y_val, = load_hole_dataset(val_idx, ds_config.ds_path, shuffle=True, split=None,
                                      normalization=md_config.normalization, one_hot=False)
    x_test, y_test, = load_hole_dataset(test_idx, ds_config.ds_path, shuffle=True, split=None,
                                        normalization=md_config.normalization, one_hot=False)
    assert isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray)
    
    # build model
    
    model = build_model(model_name=md_config.name, nb_classes=ds_config.nb_class, Chans=ds_config.nb_channel,
                        Samples=ds_config.nb_samplepoint, dropoutRate=md_config.dropoutRate)
    model.summary()
    history = train_model([x_train, y_train], [x_val, y_val], test_dataset=None, model=model,
                          model_path=md_config.ckpt_dir, batch_size=md_config.batch_size, epochs=md_config.epochs,
                          patience=md_config.earlystop_patience)
    # train_acc, train_loss = history['train_acc'], history['train_loss']
    # val_acc, val_loss = history['val_acc'], history['val_loss']
    #
    # calculate the final result
    # model = keras.models.load_model(md_config.ckpt_dir)
    # _, f_train_acc = model.evaluate(x_train, y_train)
    # _, f_test_acc = model.evaluate(x_val, y_val)
    # print(f_train_acc, f_test_acc)
    
    train_acc, val_acc, test_acc = history['train_acc'], history['val_acc'], history['test_acc']
    
    # plot the curve of results
    title = 'Weighted Acc of {} Training ({:.1f}%) & Val ({:.1f}%) & Test ({:.1f}%)'.format(
        md_config.name, train_acc[-md_config.earlystop_patience] * 100,
                        val_acc[-md_config.earlystop_patience] * 100, test_acc[-md_config.earlystop_patience] * 100)
    curve_name = ['Training acc', 'Val acc', 'Test acc', ]
    curve_data = [train_acc, val_acc, test_acc]
    color = ['r', 'g', 'b', ]
    utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, y_lim=(0, 1))
    K.clear_session()
