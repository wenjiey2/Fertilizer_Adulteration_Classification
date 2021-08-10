import json
from PIL import Image
import cv2
import numpy as np

with open('data-5000.json') as f:
    data = json.load(f)
data = data.replace("\'", "\"")
data = json.loads(data)
keys = data.keys()

# read image file name & coordinates for cropping
for key in keys:
    img_path = 'D:\\github\\YOLOv4-darknet\\images\\' + key
    out_path = 'D:\\github\\YOLOv4-darknet\\cropped\\' + key
    img = cv2.imread(img_path)
    dim = img.shape
    idx = 0
    box_num = len(data[key])
    if box_num == 0:
        print(cv2.imwrite(out_path, img))
        continue
    elif box_num > 1:
        conf_arr = []
        for i in range(box_num):
            conf_arr.append(data[key][i]['confidence'])
        idx = [np.argmax(conf_arr)]
        idx = idx[0]
    x1 = float(data[key][idx]['x1'])
    x2 = float(data[key][idx]['x2'])
    y1 = float(data[key][idx]['y1'])
    y2 = float(data[key][idx]['y2'])
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > dim[1]: x2 = dim[1]
    if y2 > dim[0]: y2 = dim[0]
    img = img[int(y1):int(y2)+1, int(x1):int(x2)+1]
    print(cv2.imwrite(out_path, img))
