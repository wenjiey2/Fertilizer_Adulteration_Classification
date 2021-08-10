import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
import pruned_model as pm
import tempfile
#tf.config.list_physical_devices('GPU')

os.environ['TF2_BEHAVIOR'] = '1'

class MyPruneQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, weight_bits=8, activation_bits=8, per_axis=False):
        super(MyPruneQuantizeConfig, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.per_axis = per_axis

    def get_weights_and_quantizers(self, layer):
        qt = tfmot.quantization.keras.quantizers
        if isinstance(layer, tf.keras.layers.SeparableConv2D):
            return [(layer.depthwise_kernel, qt.LastValueQuantizer(num_bits=self.weight_bits,
                                                                   symmetric=False,
                                                                   narrow_range=False,
                                                                   per_axis=self.per_axis)),
                    (layer.pointwise_kernel, qt.LastValueQuantizer(num_bits=self.weight_bits,
                                                                   symmetric=False,
                                                                   narrow_range=False,
                                                                   per_axis=self.per_axis))]
        return [(layer.kernel, qt.LastValueQuantizer(num_bits=self.weight_bits,
                                                               symmetric=False,
                                                               narrow_range=False,
                                                               per_axis=self.per_axis))]

    def get_activations_and_quantizers(self, layer):
        qt = tfmot.quantization.keras.quantizers
        return [(layer.activation, qt.MovingAverageQuantizer(num_bits=self.activation_bits,
                                                                       symmetric=False,
                                                                       narrow_range=False,
                                                                       per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        if isinstance(layer, tf.keras.layers.SeparableConv2D):
          layer.depthwise_kernel = quantize_weights[0]
          layer.pointwise_kernel = quantize_weights[1]
        else:
          layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}

class MyOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, output_bits=8):
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

# custom quantization layers
def apply_quantization_to_custom_layers(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        if layer.output_shape[-2] == 20:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(6, 8, True))
        elif layer.output_shape[-2] == 10:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(4, 6, True))
        elif layer.output_shape[-2] == 5:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(4, 5, True))
        elif layer.output_shape[-2] <= 80:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(8, 8, True))
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(8, 16, True))
    elif isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(5))
    elif "mish_" in layer.name:
        if int(layer.name[layer.name.find('_') + 1: ]) >= 40:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(4))
        elif int(layer.name[layer.name.find('_') + 1: ]) > 21:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(6))
        elif int(layer.name[layer.name.find('_') + 1: ]) > 10:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(8))
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(16))
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        if layer.output_shape[-2] == 40:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(8))
        elif layer.output_shape[-2] == 20:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(6))
        elif layer.output_shape[-2] == 10 or layer.output_shape[-2] == 5:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(4))
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyOpQuantizeConfig(16))
    elif isinstance(layer, tf.keras.layers.SeparableConv2D):
        if layer.output_shape[-2] == 20:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(6, 8, True))
        elif layer.output_shape[-2] == 10 or layer.output_shape[-2] == 5:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(4, 8, True))
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, MyPruneQuantizeConfig(8, 16, True))
    return layer

def main():
    # Start k-fold cross vailidation
    k = 6
    predictions = {}
    val_acc = 0
    val_loss = 0
    model = None
    select = 1

    fill_modes = ["constant", "reflect", "nearest"]
    rr = [10.0, 15.0, 10.0]
    sr = [10.0, 20.0, 10.0]
    cval = [np.random.random()*255, 0, 0]
    br = [(0.6, 1.2), (0, 0)]
    zr =  [(0.8, 1.4), (0, 0)]

    for i in range(3):
        prefix = 'dataset/'
        # prefix = 'ks\\k' + str(i) + '\\'
        # img_width, img_height = 100, 100
        train_data_dir = prefix + 'train/'
        validation_data_dir = prefix + 'test/'
        epochs = 20
        batch_size = 30

        aug_idx = -1;
        bzcwh_idx = -1

        if select == 0:
            select = 1
        elif select == 1:
            select = 2
        elif select == 2:
            select = 0
        print(fill_modes[select])

        # Generate Tensor for input images
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.7, 1.2),
            rotation_range=rr[select],
            shear_range=sr[select],
            zoom_range=(0.8, 1.3),
            channel_shift_range=60,
            fill_mode = fill_modes[select],
            cval=cval[select],
            horizontal_flip=True,
            vertical_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                          brightness_range=(0.9, 1.1),
                                          zoom_range=(1.0, 1.2),
                                          channel_shift_range=30,
                                          horizontal_flip=True)

        # Generate Iterator flow from directory
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(320, 320),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(320, 320),
            batch_size=batch_size,
            class_mode='binary')

        # Get input image dimension
        unit_image = train_generator.next()[0]
        shape = (unit_image.shape[1], unit_image.shape[2], unit_image.shape[3])

        # pruning & quantization specifications
        poly_decay = tfmot.sparsity.keras.PolynomialDecay
        prune = tfmot.sparsity.keras.prune_low_magnitude
        start_step = 101
        end_step = 584 * epochs
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        quantize_scope = tfmot.quantization.keras.quantize_scope

        # BP
        lr_schedule = pm.CustomSchedule(initial_learning_rate=0.00005/(i+1),
                                        decay_steps=584,
                                        decay_rate=0.95,
                                        warmup_steps=5840*int((i+1)/2))
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # Quantized DNN
        if not os.path.exists("quantized_model.h5"):
            base_model = keras.models.load_model("pruned_model_stripped.h5",
                                                 custom_objects={'Mish': pm.Mish, 'Sigmoid': pm.Sigmoid},
                                                 compile=False)
            quantized_model = tf.keras.models.clone_model(base_model, clone_function=apply_quantization_to_custom_layers)
            with quantize_scope({'Mish': pm.Mish,
                                 'MyOpQuantizeConfig': MyOpQuantizeConfig,
                                 'MyPruneQuantizeConfig': MyPruneQuantizeConfig,
                                 'Sigmoid': pm.Sigmoid}):
                quantized_model = tfmot.quantization.keras.quantize_apply(quantized_model)
        else:
            print("Transfer learning continued")
        quantized_model.summary()
        quantized_model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])

        # callbacks
        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            pm.MyThresholdCallback(tr_threshold=0.99, val_threshold=0.995),
            # keras.callbacks.TensorBoard(log_dir="stats\\quantize_fold"+str(i),
            #                             histogram_freq=0,
            #                             write_graph=True,
            #                             write_images=False,
            #                             update_freq="epoch",
            #                             profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='quantized_model.h5',
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
        ]
        history = quantized_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)
        quantized_model.save_weights("quantized_model.h5")

        # Create float TFLite model.
        # converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # quantized_tflite_model = converter.convert()
        # float_converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
        # float_tflite_model = float_converter.convert()

        print('Finished part ' + str(i + 1))

if __name__ == '__main__':
    main()
