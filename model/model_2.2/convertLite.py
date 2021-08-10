import tensorflow as tf
from tensorflow import keras
import pruned_model as pm
# convert to tflite
base_model = keras.models.load_model("pruned_fnf_model_2_stripped.h5",
                                        custom_objects={'Mish': pm.Mish, 'Sigmoid': pm.Sigmoid},
                                        compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
tflite_model = converter.convert()
open("pruned_fnf_model.tflite", "wb").write(tflite_model)
