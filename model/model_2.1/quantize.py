import tensorflow as tf
from tensorflow import keras
from absl import app, flags
from absl.flags import FLAGS
import pruned_model as pm
import quantized_model as qm
import tensorflow_model_optimization as tfmot
import glob
import cv2
import numpy as np

flags.DEFINE_string('precision', 'int8', 'precision (int8, fl16)')
flags.DEFINE_string('model', 'clustered_model.h5', 'model path')

def rep_data_gen():
    a = []

    for i in glob.glob('dataset\\test\\clean\\*'):
        img = cv2.imread(i)
        img = img / 255.0
        img = img.astype(np.float32)
        a.append(img)

    for i in glob.glob('dataset\\test\\adulterated\\*'):
        img = cv2.imread(i)
        img = img / 255.0
        img = img.astype(np.float32)
        a.append(img)

    a = np.array(a)
    print(a.shape) # a is np array of 2048 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(len(a)):
        yield [i]

def quantize():
    # keras_model = tf.keras.models.load_model(FLAGS.model, custom_objects={"mish":mish}, compile=False)
    with tfmot.quantization.keras.quantize_scope({'Mish': pm.Mish,
                                                  'MyPruneQuantizeConfig': qm.MyPruneQuantizeConfig,
                                                  'MyOpQuantizeConfig': qm.MyOpQuantizeConfig,
                                                  'Sigmoid': pm.Sigmoid}):
        quantized_model = keras.models.load_model(FLAGS.model, custom_objects={'Mish': pm.Mish,
                                                                     'MyPruneQuantizeConfig': qm.MyPruneQuantizeConfig,
                                                                     'MyOpQuantizeConfig': qm.MyOpQuantizeConfig,
                                                                     'Sigmoid': pm.Sigmoid}, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
    converter.experimental_new_converter = True

    if FLAGS.precision == 'fl16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
    elif FLAGS.precision == 'int8':
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.allow_custom_ops = True
        converter.representative_dataset=rep_data_gen

    tflite_model = converter.convert()
    open(FLAGS.model[: -3] + '_' + FLAGS.precision + '.tflite', 'wb').write(tflite_model)

def main(_argv):
  quantize()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
