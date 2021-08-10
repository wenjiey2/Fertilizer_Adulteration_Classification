import tensorflow as tf
from tensorflow import keras
import glob
import numpy as np
import preprocess as pp
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS
import time

flags.DEFINE_boolean('cropped', None, 'compute prediction accuracy on cropped (True) or uncropped (False) images')
flags.DEFINE_string('model', 'clustered_model.tflite', 'model path')

def main(_argv):
    field_predictions = {}
    total = 0
    correct = 0
    count = 0
    conf_count = [{}, {}, {}]
    dir = 'dataset\\test\\*'
    print(FLAGS.cropped)
    if FLAGS.cropped:
        dir = 'cropped-int8'

    # Uncomment line 23-24 below for cropped image partition before inference
    # for filename in glob.glob(dir + '\\*.jpg'):
    #      pp.preprocess(filename, 'cropped-int8\\')

    # Uncomment line 30 to 86 for inference
    for file in glob.glob(dir + '\\*.jpg'):
        field_predictions[file[file.rfind('\\') + 1 : file.rfind('_')]] = 0
        for i in range(3):
            conf_count[i][file[file.rfind('\\') + 1 : file.rfind('_')]] = 0

    conf = 0

    for file in glob.glob(dir + '\\*.jpg'):
        image = cv2.imread('D:\\github\\Fertillizer_Adulteration_Detection_app\\model\\model_2.1\\' + file)
        image = cv2.resize(image, (320, 320))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_idx = file[file.rfind('_') + 1 : file.rfind('.')]
        img_name = file[file.rfind('\\') + 1 : file.rfind('_')]
        # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image = image / 255.
        images_data = []
        for i in range(1):
            images_data.append(image)
        images_data = np.asarray(images_data).astype(np.float32)

        interpreter = tf.lite.Interpreter(model_path=FLAGS.model)
        interpreter.allocate_tensors()
        # start = time.time()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #print(input_details)
        #print(output_details)
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(time.time()-start)
        # print(output_data[0][0], file)
        if output_data[0][0] >= 0.5:
            count += 1
            conf_count[2][img_name] += 1
        if output_data[0][0] >= 0.66:
            conf_count[0][img_name] += 1
            if output_data[0][0] >= 0.83:
                conf_count[1][img_name] += 1
        if img_idx == 0 or img_idx == 3 or img_idx == 8 or img_idx == 11:
            field_predictions[img_name] += output_data[0][0]*0.071
        elif img_idx == 5 or img_idx == 6:
            field_predictions[img_name] += output_data[0][0]*0.094
        else:
            field_predictions[img_name] += output_data[0][0]*0.088
        #print(field_predictions)

    # print(conf_count)

    for key in field_predictions.keys():
        if field_predictions[key] < 0.5:
            if conf_count[0][key] >= 8:
                field_predictions[key] = 0.5
            elif conf_count[1][key] >= 2 and conf_count[2][key] >= 6:
                field_predictions[key] = 0.5
            elif conf_count[1][key] >= 1 and conf_count[0][key] >= 4 and conf_count[2][key] >= 6:
                field_predictions[key] = 0.5

    with open('test_dataset.csv', 'w') as f:
        for key in field_predictions.keys():
            total += 1
            # print(key, field_predictions[key])
            if field_predictions[key] >= 0.47:
                correct += 1
            f.write("%s,%s\n" % (key, field_predictions[key]))
    print(correct, "out of", total, "(", correct/total*100, "% ) images are correctly identified as pure.")
    print("Total counts of correctly identified pure subimages:", count)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
