import tensorflow as tf
from tensorflow import keras
import glob
import numpy as np
import preprocess as pp
from PIL import Image

def main():
    field_predictions = {}
    total = 0
    correct = 0
    model = keras.models.load_model('new_model.h5')
    # for filename in glob.glob('Urea_ROI/*.jpg'):
    #     pp.preprocess(filename, 'field/')
    for file in glob.glob('field/*.jpg'):
        field_predictions[file[file.find('/')+1 : file.find('_')]] = 0
    for file in glob.glob('field/*.jpg'):
        prediction = model.predict(load(file))[0][0]
        if prediction > 0.5:
            field_predictions[file[file.find('/')+1 : file.find('_')]] += 1
            #print(field_predictions)

    with open('test.csv', 'w') as f:
        for key in field_predictions.keys():
            total += 1
            if field_predictions[key] >= 7:
                correct += 1
            f.write("%s,%s\n" % (key, field_predictions[key]))
    print(correct, total, correct/total)

def load(file):
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

if __name__ == "__main__":
    main()
