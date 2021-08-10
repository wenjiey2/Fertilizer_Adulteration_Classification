import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs as d8
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate
#tf.config.list_physical_devices('GPU')

# mish Activation
class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * keras.backend.tanh(keras.backend.log(keras.backend.exp(inputs)+1))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

class Sigmoid(Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return keras.backend.sigmoid(inputs)

    def get_config(self):
        base_config = super(Sigmoid, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

# customize DNN modules
def backbone(input, quantize_annotate_layer):
    conv = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)

    res = keras.layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same')(bn)
    bn_res = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(res)

    dws = keras.layers.SeparableConv2D(32, (3, 3), padding='same')(bn)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    dws = keras.layers.SeparableConv2D(32, (3, 3), padding='same')(bn)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)

    mp = keras.layers.MaxPooling2D((2, 2))(bn)
    mp = keras.layers.add([mp, bn_res])
    mp = quantize_annotate_layer(Mish(), MyOpQuantizeConfig())(mp)

    conv = quantize_annotate_layer(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(mp)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv)
    conv = quantize_annotate_layer(keras.layers.Conv2D(128, (3, 3), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv)

    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(mp)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(64, (5, 1), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(64, (1, 5), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(128, (3, 3), strides=1, padding='same'), MyPruneQuantizeConfig(24, 30, True, False))(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(24))(conv_1)

    concat = tf.keras.layers.concatenate(inputs=[bn, bn_1], axis=3)
    return concat

def neck_module1(ip, end_step, prune, decay, quantize_annotate_layer):
    input = quantize_annotate_layer(Mish(), MyOpQuantizeConfig(16))(ip)
    conv = quantize_annotate_layer(prune(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(8, 24, True, True))(input)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv)
    conv = quantize_annotate_layer(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv)
    conv = quantize_annotate_layer(keras.layers.Conv2D(64, (3, 1), strides=1, padding='same'), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv)

    conv_1 = quantize_annotate_layer(prune(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(8, 24, True, True))(input)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), d8.NoOpQuantizeConfig())(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(128, (3, 1), strides=1, padding='same'), d8.NoOpQuantizeConfig())(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_1)
    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), d8.NoOpQuantizeConfig())(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_1)
    conv_1 = quantize_annotate_layer(prune(keras.layers.Conv2D(64, (3, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(8, 24, True, True))(bn_1)
    bn_1 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_1)

    mp = keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(input)
    conv_2 = quantize_annotate_layer(prune(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(8, 24, True, True))(mp)
    bn_2 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_2)

    conv_3 = quantize_annotate_layer(prune(keras.layers.Conv2D(64, (1, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(8, 24, True, True))(input)
    bn_3 = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(12))(conv_3)

    concat = keras.layers.concatenate(inputs=[bn, bn_1, bn_2, bn_3], axis=3)
    sum = keras.layers.add([concat, input])
    return sum

def neck_module2(ip, end_step, prune, decay, quantize_annotate_layer):
    input = quantize_annotate_layer(Mish(), MyOpQuantizeConfig(8))(ip)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(128, (3, 3), padding='same'), decay(0.1, 0.2, 0, end_step)), d8.NoOpQuantizeConfig())(input)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(10))(dws)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(192, (3, 3), padding='same'), decay(0.1, 0.2, 0, end_step)), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(10))(dws)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(256, (3, 3), padding='same'), decay(0.1, 0.2, 0, end_step)), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(10))(dws)
    sum = keras.layers.add([input, bn])

    conv = quantize_annotate_layer(prune(keras.layers.Conv2D(512, (1, 1), strides=1, padding='same'), decay(0.1, 0.2, 0, end_step)), MyPruneQuantizeConfig(5, 6, True, True))(sum)

    dws = quantize_annotate_layer(keras.layers.SeparableConv2D(256, (3, 3), padding='same'), d8.NoOpQuantizeConfig())(sum)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(8))(dws)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(384, (3, 3), padding='same'), decay(0.1, 0.2, 0, end_step)), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(8))(dws)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(512, (3, 3), padding='same'), decay(0.1, 0.2, 0, end_step)), d8.NoOpQuantizeConfig())(bn)
    bn = quantize_annotate_layer(keras.layers.BatchNormalization(axis=-1, momentum=0.9), MyOpQuantizeConfig(8))(dws)
    sum = keras.layers.add([conv, bn])
    return sum

def neck_module3(ip, end_step, prune, decay, quantize_annotate_layer):
    input = quantize_annotate_layer(Mish(), MyOpQuantizeConfig(7))(ip)
    conv = quantize_annotate_layer(keras.layers.Conv2D(128, (1, 1), strides=1, padding='same'), MyPruneQuantizeConfig(4, 6, True, False))(input)
    bn = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv)
    conv = quantize_annotate_layer(prune(keras.layers.Conv2D(192, (1, 3), strides=1, padding='same'), decay(0.4, 0.6, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn)
    bn = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv)
    conv = quantize_annotate_layer(prune(keras.layers.Conv2D(256, (3, 1), strides=1, padding='same'), decay(0.4, 0.6, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn)
    bn = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv)

    conv_01 = quantize_annotate_layer(prune(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), decay(0.5, 0.7, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn)
    bn_01 = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv_01)
    conv_02 = quantize_annotate_layer(prune(keras.layers.Conv2D(128, (3, 1), strides=1, padding='same'), decay(0.6, 0.8, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn)
    bn_02 = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv_02)

    conv_1 = quantize_annotate_layer(keras.layers.Conv2D(256, (1, 1), strides=1, padding='same'), MyPruneQuantizeConfig(4, 6, True, False))(input)
    bn_1 = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv_1)

    conv_11 = quantize_annotate_layer(prune(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), decay(0.6, 0.8, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn_1)
    bn_11 = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv_11)
    conv_12 = quantize_annotate_layer(prune(keras.layers.Conv2D(128, (3, 1), strides=1, padding='same'), decay(0.4, 0.6, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(bn_1)
    bn_12 = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv_12)

    concat = keras.layers.concatenate(inputs=[bn_01, bn_02, bn_11, bn_12], axis=3)
    sum = keras.layers.add([input, concat])

    dws = quantize_annotate_layer(keras.layers.SeparableConv2D(512, (3, 3), padding='same'), d8.NoOpQuantizeConfig())(input)
    bn = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(dws)

    concat = keras.layers.concatenate(inputs=[sum, bn], axis=3)
    return concat

def head(ip, end_step, prune, decay, quantize_annotate_layer):
    input = quantize_annotate_layer(Mish(), MyOpQuantizeConfig(6))(ip)
    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(512, (3, 3), padding='same'), decay(0.7, 0.85, 0, end_step)), d8.NoOpQuantizeConfig())(input)
    bn_dws = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9),decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(dws)
    conv = quantize_annotate_layer(prune(keras.layers.Conv2D(512, (1, 1), strides=1, padding='same'), decay(0.4, 0.6, 0, end_step)), MyPruneQuantizeConfig(4, 5, True, True))(input)
    bn = quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(conv)
    sum = keras.layers.add([bn, bn_dws])

    dws = quantize_annotate_layer(prune(keras.layers.SeparableConv2D(1024, (3, 3), padding='same'), decay(0.5, 0.75, 0, end_step)), d8.NoOpQuantizeConfig())(sum)
    bn =  quantize_annotate_layer(prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step)), MyOpQuantizeConfig(6))(dws)
    mish_act = quantize_annotate_layer(Mish(), MyOpQuantizeConfig(5))(bn)
    gap = keras.layers.GlobalAveragePooling2D()(mish_act)
    return gap

def FADNet(input_shape, end_step, prune, decay, quantize_annotate_layer):
    img_input = keras.layers.Input(shape=input_shape)
    bb = backbone(img_input, quantize_annotate_layer)
    n1 = neck_module1(bb, end_step, prune, decay, quantize_annotate_layer)
    mp1 = keras.layers.MaxPooling2D((2, 2))(n1)
    n2 = neck_module2(mp1, end_step, prune, decay, quantize_annotate_layer)
    mp2 = keras.layers.MaxPooling2D((2, 2))(n2)
    n3 = neck_module3(mp2, end_step, prune, decay, quantize_annotate_layer)
    mp3 = keras.layers.MaxPooling2D((2, 2))(n3)
    h = head(mp3, end_step, prune, decay, quantize_annotate_layer)
    fl = keras.layers.Flatten()(h)
    d2 = quantize_annotate_layer(prune(keras.layers.Dense(1), decay(0.6, 0.8, 0, end_step)), MyPruneQuantizeConfig(4, 5, False, True))(fl)
    d2 = quantize_annotate_layer(Sigmoid(), MyOpQuantizeConfig(7))(d2)
    model = tf.keras.Model(img_input, d2)
    return model

# Customize early stopping callback
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, tr_threshold, val_threshold):
        super(MyThresholdCallback, self).__init__()
        self.tr_threshold = tr_threshold
        self.val_threshold = val_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        tr_acc = logs["accuracy"]
        if val_acc >= self.val_threshold and tr_acc >= self.tr_threshold:
            self.model.stop_training = True

# Customize learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = self.initial_learning_rate * (self.decay_rate**((step-self.warmup_steps)/self.decay_steps))
        arg2 = self.initial_learning_rate * step / self.warmup_steps
        return tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "warmup_steps": self.warmup_steps,
        }

# custom pruning layers
def apply_pruning_to_custom_layers(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        if layer.output_shape[-1] == 128 and layer.output_shape[-2] == 6:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.1, 0.3, 2610, 26100))
        elif layer.output_shape[-1] == 512 or layer.name == 'prune_low_magnitude_conv2d_22':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.3, 0.6, 2610, 26100))
        elif layer.name == 'prune_low_magnitude_conv2d_23' or layer.name == 'prune_low_magnitude_conv2d_25':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.5, 0.8, 2610, 26100))
        elif layer.name == 'prune_low_magnitude_conv2d_26':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.3, 0.6, 2610, 26100))
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        if layer.output_shape[-2] == 3 or layer.output_shape[-2] == 6:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.7, 0.9, 2610, 26100))
    elif isinstance(layer, tf.keras.layers.SeparableConv2D):
        if layer.output_shape[-2] == 12:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.1, 0.2, 2610, 26100))
        elif layer.output_shape[-2] == 3:
            if layer.output_shape[-1] == 1024:
                return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.4, 0.6, 2610, 26100))
            elif layer.output_shape[-1] == 512:
                return tfmot.sparsity.keras.prune_low_magnitude(layer, tfmot.sparsity.keras.PolynomialDecay(0.6, 0.8, 2610, 26100))
    return layer

# custom quantization config
class MyPruneQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, weight_bits=32, activation_bits=32, per_axis=False, prune=True):
        super(MyPruneQuantizeConfig, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.per_axis = per_axis
        self.prune = prune

    def get_weights_and_quantizers(self, layer):
        qt = tfmot.quantization.keras.quantizers
        if not isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            self.prune = False
        if self.prune:
            return [(layer.layer.kernel, qt.LastValueQuantizer(num_bits=self.weight_bits,
                                                               symmetric=False,
                                                               narrow_range=False,
                                                               per_axis=self.per_axis))]
        else:
            return [(layer.kernel, qt.LastValueQuantizer(num_bits=self.weight_bits,
                                                     symmetric=False,
                                                     narrow_range=False,
                                                     per_axis=self.per_axis))]

    def get_activations_and_quantizers(self, layer):
        qt = tfmot.quantization.keras.quantizers
        if not isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            self.prune = False
        if self.prune:
            return [(layer.layer.activation, qt.MovingAverageQuantizer(num_bits=self.activation_bits,
                                                                       symmetric=False,
                                                                       narrow_range=False,
                                                                       per_axis=False))]
        else:
            return [(layer.activation, qt.MovingAverageQuantizer(num_bits=self.activation_bits,
                                                             symmetric=False,
                                                             narrow_range=False,
                                                             per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        if not isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            self.prune = False
        if self.prune:
            layer.layer.kernel = quantize_weights[0]
        else:
            layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        if not isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            self.prune = False
        if self.prune:
            layer.layer.activation = quantize_activations[0]
        else:
            layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

class MyOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, output_bits=32):
        super(MyOpQuantizeConfig, self).__init__()
        self.output_bits = output_bits

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.output_bits,
                                                                           symmetric=False,
                                                                           narrow_range=False,
                                                                           per_axis=False)]

    def get_config(self):
        return {}
# Uncomment the code below to see compressed size after pruning
# def get_gzipped_model_size(file):
#   # Returns size of gzipped model, in bytes.
#   import os
#   import zipfile
#
#   _, zipped_file = tempfile.mkstemp('.zip')
#   with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
#     f.write(file)
#
#   return os.path.getsize(zipped_file)

def main():
    # Start k-fold cross vailidation
    k = 6
    predictions = {}
    val_acc = 0
    val_loss = 0
    model = None
    select = 1

    # Uncomment this to visualize an example of the learn rate schedule
    # temp_learning_rate_schedule = CustomSchedule(initial_learning_rate=0.001*0.95**50, decay_steps=522, decay_rate=0.95, warmup_steps=0)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()

    fill_modes = ["constant", "reflect", "nearest"]

    for i in range(k):
        prefix = 'ks\\k' + str(i) + '\\'
        img_width, img_height = 100, 100
        train_data_dir = prefix + 'train\\'
        validation_data_dir = prefix + 'test\\'
        epochs = 50 + 10*i
        batch_size = 30

        if select == 0:
            select = np.random.randint(1, 3)
        else:
            select = 0
        print(fill_modes[select])

        # Generate Tensor for input images
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(0.8, 1.2),
            rotation_range=0.1,
            shear_range=0.2,
            zoom_range=0.25,
            channel_shift_range=30,
            fill_mode = fill_modes[select],
            cval=0.0,
            horizontal_flip=True,
            vertical_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Generate Iterator flow from directory
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        # Get input image dimension
        unit_image = train_generator.next()[0]
        shape = (unit_image.shape[1], unit_image.shape[2], unit_image.shape[3])

        # pruning & quantization specifications
        poly_decay = tfmot.sparsity.keras.PolynomialDecay
        prune = tfmot.sparsity.keras.prune_low_magnitude
        end_step = 522 * epochs
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        quantize_scope = tfmot.quantization.keras.quantize_scope

        # Pruned DNN
        pruned_model = FADNet(shape, end_step, prune, poly_decay, quantize_annotate_layer)

        # Quantized DNN
        annotated_model = quantize_annotate_model(pruned_model)
        with quantize_scope({'MyPruneQuantizeConfig': MyPruneQuantizeConfig,
                             'MyOpQuantizeConfig': MyOpQuantizeConfig,
                             'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                             'Mish': Mish,
                             'Sigmoid': Sigmoid}):
            quantized_model = tfmot.quantization.keras.quantize_apply(annotated_model)
            quantized_model.summary()

        # BP
        th = 0.97
        decay_exp = 0
        if i > 0:
            decay_exp = epochs - 10
            th = 0.99
        if i == 5:
            th = 0.985

        lr_schedule = CustomSchedule(initial_learning_rate=0.0005*(0.95**decay_exp)/(i+1), decay_steps=522, decay_rate=0.95, warmup_steps=5220*int((i+1)/2))
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # check for transfer learning
        if os.path.exists("new_model.h5"):
            base_model = keras.models.load_model("new_model.h5", custom_objects={'Mish': Mish,
                                                                                 'QuantizeAnnotate':quantize_annotate.QuantizeAnnotate,
                                                                                 'MyPruneQuantizeConfig': MyPruneQuantizeConfig,
                                                                                 'MyOpQuantizeConfig': MyOpQuantizeConfig,
                                                                                 'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                                                                                 'NoOpQuantizeConfig': d8.NoOpQuantizeConfig,
                                                                                 'Sigmoid': Sigmoid}, compile=False)
            pruned_model = tf.keras.models.clone_model(base_model, clone_function=apply_pruning_to_custom_layers)
            print("Transfer learning continued")

        pruned_model.summary()
        pruned_model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])

        # callbacks
        my_callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20+5*i, restore_best_weights=True),
            MyThresholdCallback(tr_threshold=0.94, val_threshold=th),
            #keras.callbacks.TensorBoard(log_dir="stats\\fold"+str(i), histogram_freq=0, write_graph=True, write_images=False, update_freq="epoch", profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='new_model.h5', save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
        ]
        history = pruned_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)

        model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

        # visualize compression
        # _, pruned_keras_file = tempfile.mkstemp('.h5')
        # tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        # print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

        model_for_export.save("new_model.h5")
        print('Finished part ' + str(i + 1))


if __name__ == '__main__':
    main()
