import cv2
import numpy as np
from common import fit_weight_normalizer, images_generator, getInterestRegionMask, mkdir_p, get_partition, normalize_brightness_bgr
from settings import CAMERA_OFFSET, CROP_TOP, CROP_BOTTOM, NORMALIZE_BRIGHTNESS
from keras.models import load_model
import sys

interestRegionMask = None

weight_normalizer = fit_weight_normalizer()

partition = get_partition(weight_normalizer)
for key in ["train", "validation"]:
    partition[key] = set([x["id"] for x in partition[key]])

modelFileName = sys.argv[1] if len(sys.argv) > 1 else "model.h5"
model = load_model(modelFileName)

report = open("report.csv", "w")
report.write("key,true_angle,pred_angle,usage\n")

for item in images_generator():
    side_by_side_img = []
    combined_path = None
    
    for key, offset in [('left', CAMERA_OFFSET), ('center', 0), ('right', -CAMERA_OFFSET)]:
        orig_image_bgr = cv2.imread(item[key])
        if NORMALIZE_BRIGHTNESS:
            orig_image_bgr = normalize_brightness_bgr(orig_image_bgr)
        orig_image = orig_image_bgr.astype(np.float64)[...,::-1]
        
        image = orig_image[CROP_TOP:-CROP_BOTTOM].copy()
        
        if interestRegionMask is None:
            interestRegionMask = getInterestRegionMask(image.shape[0], image.shape[1])
            interestRegionMask = np.reshape(interestRegionMask, list(interestRegionMask.shape) + [1])

        image = np.multiply(interestRegionMask, image)
        
        # line_image = np.zeros(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        
        true_angle = item['angle'] + offset
        pred_angle = float(model.predict(orig_image[None, :, :, :], batch_size=1))
        
        # draw true angle in green
        cv2.line(image, (int(width/2), height-1), (int((1 + true_angle/2.) * width/2), 0), (0, 255, 0), thickness=5)
        # draw predicted angle in red
        cv2.line(image, (int(width/2), height-1), (int((1 + pred_angle/2.) * width/2), 0), (255, 0, 0), thickness=5)

        # use green if training image, red if validation, blue if not used in training
        itemId = str(item['id']) + "_" + key
        if itemId in partition["train"]:
            usage = 'train'
            color = (0, 255, 0)
        elif itemId in partition["validation"]:
            usage = 'valid'
            color = (255, 0, 0)
        else:
            usage = 'none'
            color = (0, 0, 255)
        
        cv2.rectangle(image, (0, 0), (width-1, height-1), color)
        
        cv2.putText(
            image,
            usage.upper(),
            (width - 50, 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
        
        report.write("{key},{true_angle},{pred_angle},{usage}\n".format(**locals()))
        
        side_by_side_img.append(image)
        
        path = ("angle_overlay/" + item[key]).split("/")
        filename = path[-1].split("_")
        dirPath = "/".join(path[:-2] + [filename[0]])
        mkdir_p(dirPath)
        cv2.imwrite("/".join([dirPath, "_".join(filename[1:])]), image[...,::-1])

        if combined_path is None:
            combinedDirPath = "/".join(path[:-2] + ["combined"])
            mkdir_p(combinedDirPath)
            combined_path = "/".join([combinedDirPath, "_".join(filename[1:])])
        
    cv2.imwrite(combined_path, np.concatenate(side_by_side_img, axis=1)[...,::-1])
report.close()
