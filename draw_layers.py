import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K
from common import generator, get_partition, fit_weight_normalizer, normalize_brightness

weight_normalizer = fit_weight_normalizer()

partition = get_partition(weight_normalizer)

model = load_model("model.h5")

inp = model.input
outputs = [layer.output for layer in model.layers]
functor = K.function([inp]+ [K.learning_phase()], outputs)

batch = next(generator(partition['train']))[0]

# Testing
shape = list(batch.shape)
shape[0] = 1
for j in range(batch.shape[0]):
    test = np.reshape(batch[j], shape)
    layer_outs = functor([test, 1.])
    for i, out in enumerate(layer_outs):
        if len(out.shape) == 4 and out.shape[-1] >= 3:
            out = out[:, :, :, :3]
            out = out - np.min(out, axis=(0, 1, 2))
            out = out / np.max(out, axis=(0, 1, 2))
            out *= 255
            out = np.reshape(out, out.shape[1:]).astype(np.uint8)
            out = normalize_brightness(out)
            cv2.imwrite("layers/" + str(j + 1) + "layer_" + str(i + 1) + ".jpg", out[...,::-1])

import gc
gc.collect()
