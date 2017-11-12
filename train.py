import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(42)

import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import cv2
from common import images_generator, WeightNormalizer, get_partition, generator, fit_weight_normalizer
from settings import CAMERA_OFFSET, CROP_TOP, CROP_BOTTOM, BATCH_SIZE

import json
with open("modelConf.json", "r") as modelConf_file:
    modelConf = json.load(modelConf_file)

weight_normalizer = fit_weight_normalizer(modelConf["example_imbalance"])

partition = get_partition(weight_normalizer)
   
TRAIN_STEPS_PER_EPOCH = int(len(partition['train']) / BATCH_SIZE)
VAL_STEPS_PER_EPOCH = int(len(partition['validation']) / BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Dropout, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint

def interestRegion(img):
    from common import getInterestRegionMask
    import tensorflow as tf

    interestRegionMask = getInterestRegionMask(img.shape[1].value, img.shape[2].value)
    interestRegionMask = tf.expand_dims(tf.convert_to_tensor(interestRegionMask, np.float32), 2)
    return tf.multiply(interestRegionMask, img)

model = Sequential()
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(interestRegion))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
if modelConf.get("convo_1", True):
    model.add(Conv2D(3 * modelConf["convo_depth_factor"], (5, 5), activation='relu'))
if modelConf.get("convo_2", True):
    model.add(Conv2D(5 * modelConf["convo_depth_factor"], (5, 5), activation='relu'))
if modelConf.get("convo_3", True):
    model.add(Conv2D(7 * modelConf["convo_depth_factor"], (5, 5), activation='relu'))
if modelConf.get("convo_4", True):
    model.add(Conv2D(9 * modelConf["convo_depth_factor"], (5, 5), activation='relu'))
if modelConf.get("convo_5", True):
    model.add(Conv2D(11 * modelConf["convo_depth_factor"], (3, 3), activation='relu'))
if modelConf["add_pooling"]:
    model.add(MaxPooling2D())
model.add(Flatten())
if modelConf["dropout_1"]:
    model.add(Dropout(0.5))
model.add(Dense(24 * modelConf["dense_1_factor"], activation='relu'))
if modelConf["dropout_2"]:
    model.add(Dropout(0.5))
model.add(Dense(12 * modelConf["dense_2_factor"], activation='relu'))
if modelConf["dense_3"]:
    if modelConf["dropout_3"]:
        model.add(Dropout(0.5))
    model.add(Dense(6 * modelConf["dense_3_factor"], activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(
    generator(partition['train']),
    TRAIN_STEPS_PER_EPOCH,
    epochs=15,
    verbose=2,
    validation_data=generator(partition['validation']),
    validation_steps=VAL_STEPS_PER_EPOCH,
    callbacks = [
#         ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    ]
)

model.save('model.h5')

modelConf["history"] = history.history
import uuid
unique_filename = str(uuid.uuid4())
with open(os.path.join("history", unique_filename), "w") as history_file:
    json.dump(modelConf, history_file)

import gc
gc.collect()

