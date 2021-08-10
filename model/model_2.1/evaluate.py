import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import csv
import glob
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from predict import sub_predict
import pruned_model as pm
import quantized_model as qm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

flags.DEFINE_string('precision', 'fl32', 'precision (int8, fl16, fl32)')
flags.DEFINE_string('model', 'new_model.tflite', 'model path')

# Helper function that loads images
def load(file):
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def sub_predict_tflite(prefix, qmodel):
    predictions = {}
    adulterates = sorted(glob.glob(prefix + 'test\\adulterated\\*.jpg'))
    cleans = sorted(glob.glob(prefix + 'test\\clean\\*.jpg'))

    interpreter = tf.lite.Interpreter(model_path=qmodel)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pics = int(len(adulterates) / 12)
    for p in range(pics):
        acc = 0
        cropped = adulterates[p * 12: (p + 1) * 12]
        for piece in cropped:
            interpreter.set_tensor(input_details[0]['index'], load(piece))
            all_layers_details = interpreter.get_tensor_details()
            interpreter.invoke()
            # for layer in all_layers_details:
            #     if layer['index'] < 20:
            #         print(str(layer['index']))
            #         print(layer['name'])
            #         print(layer['shape'])
            #         print(interpreter.get_tensor(layer['index']))
            #         print(interpreter.get_tensor(layer['index']).dtype)
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            if prediction >= 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('\\') + 1: filename.rfind('_')]
        predictions[filename] = acc

    pics = int(len(cleans) / 12)
    for p in range(pics):
        acc = 0
        cropped = cleans[p * 12: (p + 1) * 12]
        for piece in cropped:
            interpreter.set_tensor(input_details[0]['index'], load(piece))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            if prediction >= 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('\\') + 1: filename.rfind('_')]
        predictions[filename] = acc

    # save pure prediction counts of subimages for test images
    with open('predictions_' + FLAGS.precision + '.csv', 'a') as f:
        for key in predictions.keys():
            f.write("%s,%s\n" % (key, predictions[key]))

def main(_argv):
    if FLAGS.model[-3:] == ".h5":
        model = keras.models.load_model(FLAGS.model, custom_objects={'Mish': pm.Mish,
                                                                             'MyPruneQuantizeConfig': qm.MyPruneQuantizeConfig,
                                                                             'Sigmoid': pm.Sigmoid}, compile=False)
        sub_predict('dataset\\', model)
    elif FLAGS.model[-7:] == ".tflite":
        sub_predict_tflite('dataset\\', FLAGS.model)

    print("Finished subimage prediction")

    # predictions = {}
    # with open('predictions_' + FLAGS.precision + '.csv', newline='\n') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     for row in reader:
    #         predictions[row[0]] = int(row[1])
    #
    # percentages = np.zeros(12)
    # for i in range(12):
    #     print('\nWhen threshold is %d: ' %(i+1), file=open("summary_"+FLAGS.precision+".txt", "a"))
    #     misclassified = 0
    #     a, b, c, d = 0, 0, 0, 0
    #     for image in predictions.keys():
    #         if image[:1] == 'a' and predictions[image] < i:
    #             a += 1
    #         elif image[:1] == 'a' and predictions[image] >= i:
    #             b += 1
    #         elif image[:1] != 'a' and predictions[image] < i:
    #             c += 1
    #         elif image[:1] != 'a' and predictions[image] >= i:
    #             d += 1
    #     percentages[i] = round((a + d) / (a + b + c + d), 3)
    #     print('%s confusion matrix: %d, %d, %d, %d' % ("Test dataset", a, b, c, d), file=open("summary_"+FLAGS.precision+".txt", "a"))
    #     misclassified += (b + c)
    #     #print(percentages[6][i])
    #     print('%d images misclassified' % misclassified, file=open("summary_"+FLAGS.precision+".txt", "a"))
    #
    # # for accuracy in percentages:
    # #     print(accuracy)
    #
    # x = np.arange(12)
    # plt.plot(x+1, percentages)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.05))
    # plt.grid()
    # plt.savefig('Percentages_' + FLAGS.precision + '.png')
    #
    #
    # opt_threshold = np.argmax(percentages)
    # print("Optimal threshold:", opt_threshold + 1, file=open("summary_" + FLAGS.precision + ".txt", "a"))
    # print("\nAccuracy for optimal threshold: ", percentages[opt_threshold], file=open("summary_" + FLAGS.precision + ".txt", "a"))

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
