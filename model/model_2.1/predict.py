import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import pruned_model as pm
import quantized_model as qm

# Helper function that loads images
def load(file):
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# predict on subimages
def sub_predict(prefix, model):
    predictions = {}
    adulterates = glob.glob(prefix + 'test\\adulterated\\*.jpg')
    pics = int(len(adulterates) / 12)
    for p in range(pics):
        cropped = adulterates[p * 12: (p + 1) * 12]
        acc = 0
        for piece in cropped:
            prediction = model.predict(load(piece))[0][0]
            if prediction >= 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('\\') + 1: filename.rfind('_')]
        predictions[filename] = acc

    cleans = glob.glob(prefix + 'test\\clean\\*.jpg')
    pics = int(len(cleans) / 12)
    for p in range(pics):
        cropped = cleans[p * 12: (p + 1) * 12]
        acc = 0
        for piece in cropped:
            prediction = model.predict(load(piece))[0][0]
            if prediction >= 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('\\') + 1: filename.rfind('_')]
        predictions[filename] = acc

    # save pure prediction counts of subimages for test images
    with open('predictions.csv', 'w') as f:
        for key in predictions.keys():
            f.write("%s,%s\n" % (key, predictions[key]))

def main():
    model = keras.models.load_model("clustered_model.h5", custom_objects={'Mish': pm.Mish,
                                                                         'MyPruneQuantizeConfig': qm.MyPruneQuantizeConfig,
                                                                         'Sigmoid': pm.Sigmoid}, compile=False)
    # predict on subimages
    for i in range(6):
        start = time.time()
        sub_predict('dataset' + '\\', model)
        print("Inference time:", time.time()-start)

if __name__ == '__main__':
    main()
