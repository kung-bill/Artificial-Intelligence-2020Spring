# import keras
from tensorflow import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from nms import nms_index, get_iou, comput_precision
# use this to change which GPU to use
gpu = "1, "

# set the modified tf session as backend in keras
setup_gpu(gpu)

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# load image
#image = read_image_bgr('000000008021.jpg')
image = read_image_bgr('test.jpg')

# copy to draw on
draw = image.copy()
draw2 = image.copy()

draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)



score_thr = 0.39
# correct for image scale
boxes /= scale

boxes = boxes[0]
scores = scores[0]
labels = labels[0]

# visualize detections
boxes_thred = []
scores_thred = []
labels_thred = []

for box, score, label in zip(boxes, scores, labels):
    # scores are sorted so we can break
    if score > score_thr:
        boxes_thred.append(box)
        scores_thred.append(score)
        labels_thred.append(label)
print(f"len(boxes_thred)={len(boxes_thred)}")

count = 0
for box, score, label in zip(boxes_thred, scores_thred, labels_thred):
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    #caption = "{} {:.3f}".format(labels_to_names[label], score)
    caption = f"{count}"
    draw_caption(draw, b, caption)
    count += 1
   

select_ind = nms_index(boxes_thred, scores_thred, 0.5)

print(f"len(select_ind)={len(select_ind)}")

picked_boxes = []
picked_score = []
picked_labels = []
for ind in select_ind:
    picked_boxes.append(boxes_thred[ind])
    picked_score.append(scores_thred[ind])
    picked_labels.append(labels_thred[ind])


draw2 = cv2.cvtColor(draw2, cv2.COLOR_BGR2RGB)

# visualize detections
count = 0
for box, score, label in zip(picked_boxes, picked_score, picked_labels):
    # scores are sorted so we can break
    if score < score_thr:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw2, b, color=color)
    
    #caption = "{} {:.3f}".format(labels_to_names[label], score)
    caption = f"{count}"
    draw_caption(draw2, b, caption)
    count += 1

iou_thr = 0.5
acc = comput_precision(boxes_thred, labels_thred, picked_boxes, picked_labels, iou_thr)
print(f"comput_precision={acc}")

cv2.imshow('before NMS (true)', draw)
cv2.imshow('after NMS (predict)', draw2)
cv2.waitKey(0)




