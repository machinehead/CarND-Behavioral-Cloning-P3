import csv
import numpy as np
import tensorflow as tf
import errno    
import os
import random
import cv2
from settings import TRAIN_SHARE, PARTITION_SEED, CAMERA_OFFSET, BATCH_SIZE, NORMALIZE_BRIGHTNESS, WEIGHT_NORMALIZER_BINS

# Normalize brightness for BGR images
def normalize_brightness_bgr(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Normalize brightness for RGB images
def normalize_brightness(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


# Generator for filenames and turn angles
def images_generator(folders=['1_2_laps_track_1', '2_sharp_turns_track_1', '3_1_lap_track_2', '4_1_lap_track_2', '5_1_lap_track_2', '6_1_lap_track_1', '7_corrective_turn_track_2', '8_next_turn_track_2', '9_edge_avoidance_track_2']):
    id = 1
    for folder in folders:
        with open(folder + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                item = {"id": id}
                for idx, key in [(0, 'center'), (1, 'left'), (2, 'right')]:
                    source_path = line[idx]
                    filename = source_path.split('/')[-1]
                    current_path = folder + '/IMG/' + filename
                    item[key] = current_path
                item['angle'] = float(line[3])
                yield item
                id += 1

# Frequency normalization for different angles; this is supposed to penalize very frequent turn angle values;
# see explore.ipynb for some demos
class WeightNormalizer:
    def __init__(self, bins, imbalance=2.):
        self.bins = bins
        self.max_weight = 50
        self.imbalance = imbalance
    
    def fit(self, angles):
        hist, edges = np.histogram(angles, bins=self.bins, density=True)
        self.hist = [x * (edges[-1] - edges[0]) / self.bins for x in hist]
        smallest_bin = min([x for x in self.hist if x > 0])
        self.bin_frequencies = [x / smallest_bin for x in self.hist]
        self.edges = edges
        return self
        
    def get_weight(self, angle):
        for i, edge in enumerate(self.edges):
            if angle < edge:
                if i == 0:
                    return self.max_weight
                else:
                    return min(1/self.hist[i-1], self.max_weight)
            elif angle == edge:
                if i == len(self.edges) - 1:
                    return self.max_weight
                else:
                    return min(1/self.hist[i], self.max_weight)
        return self.max_weight
    
    # returns relative frequency; how more frequent the current bin is compared to the lowest frequency bin
    def get_bin_frequency(self, angle):
        for i, edge in enumerate(self.edges):
            if angle < edge:
                if i == 0:
                    return 1.
                else:
                    return self.bin_frequencies[i-1]
            elif angle == edge:
                if i == len(self.edges) - 1:
                    return 1.
                else:
                    return self.bin_frequencies[i]
        return 1.
    
    def random_keep(self, angle):
        return random.random() * self.get_bin_frequency(angle) < self.imbalance

def getInterestRegionMask(height, width):
    return np.array([[1 if y > (-.5 + 0.8/width*x)*height
                        and y > (.3 - 0.8/width*x)*height
                        else 0
                      for x in range(width)] for y in range(height)])

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def get_partition(weight_normalizer):
    partition = { 'train': [], 'validation': [] }
    random.seed(PARTITION_SEED)
    for item in images_generator():
        for key, offset in [('center', 0), ('left', CAMERA_OFFSET), ('right', -CAMERA_OFFSET)]:
            newItem1 = {
                'id': str(item['id']) + "_" + key,
                'path': item[key],
                'angle': float(item['angle']) + offset,
                'flip': False
            }
            newItem2 = {
                'id': str(item['id']) + "_" + key,
                'path': item[key],
                'angle': -float(item['angle']) - offset,
                'flip': True
            }
            for newItem in [newItem1, newItem2]:
                if weight_normalizer.random_keep(newItem['angle']):
                    partition['train' if random.random() < TRAIN_SHARE else 'validation'].append(newItem)
    return partition
    
def generator(items):
    current_batch_size = 0
    images = []
    measurements = []
    weights = []

    items_shuffled = list(items)
    random.shuffle(items_shuffled)
    
    while True:
        for item in items_shuffled:
            image_bgr = cv2.imread(item['path'])
            if NORMALIZE_BRIGHTNESS:
                image_bgr = normalize_brightness_bgr(image_bgr)
            image = image_bgr[...,::-1]

            if item['flip']:
                image = np.fliplr(image)
                
            images.append(image)
            measurements.append(item['angle'])
            weights.append(1.)
                
            current_batch_size += 1
            if current_batch_size == BATCH_SIZE:
                yield (np.array(images), np.array(measurements), np.array(weights))
                current_batch_size = 0
                images = []
                measurements = []
                weights = []
                
def fit_weight_normalizer(imbalance=2.):
    angles = []
    for item in images_generator():
        angles.append(item['angle'])
        angles.append(-item['angle'])
        angles.append(item['angle'] + CAMERA_OFFSET)
        angles.append(-item['angle'] - CAMERA_OFFSET)
        angles.append(item['angle'] - CAMERA_OFFSET)
        angles.append(-item['angle'] + CAMERA_OFFSET)

    return WeightNormalizer(WEIGHT_NORMALIZER_BINS, imbalance).fit(angles)

