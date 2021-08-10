import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import TensorBoard
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate
#tf.config.list_physical_devices('GPU')

# mish Activation
class Mish(Layer):
    '''
    Mish Activation Function.
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
def backbone(input, start_step, end_step, prune, decay):
    conv = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    res = prune(keras.layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same'), decay(0.02, 0.08, start_step, end_step))(act)
    bn_res = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.05, 0.15, start_step, end_step))(res)
    act_res = Mish()(bn_res)

    dws = prune(keras.layers.SeparableConv2D(16, (3, 3), padding='same'), decay(0.02, 0.08, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.05, 0.15, start_step, end_step))(dws)
    act = Mish()(bn)
    dws = prune(keras.layers.SeparableConv2D(16, (3, 3), padding='same'), decay(0.05, 0.15, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.1, 0.2, start_step, end_step))(dws)
    act = Mish()(bn)

    mp = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(act)
    mp = keras.layers.add([mp, act_res])
    mp = Mish()(mp)

    conv = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(mp)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv)
    act = Mish()(bn)
    conv = prune(keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(bn)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv)
    act = Mish()(bn)

    conv_1 = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(mp)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(32, (5, 1), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(32, (1, 5), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'), decay(0.15, 0.3, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.2, 0.4, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)

    concat = tf.keras.layers.concatenate(inputs=[act, act_1], axis=3)
    return concat

def neck_module1(ip, start_step, end_step, prune, decay):
    input = Mish()(ip)
    conv = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.2, 0.4, start_step, end_step))(input)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.4, 0.6, start_step, end_step))(conv)
    act = Mish()(bn)
    conv = prune(keras.layers.Conv2D(64, (1, 3), strides=1, padding='same'), decay(0.2, 0.4, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.4, 0.6, start_step, end_step))(conv)
    act = Mish()(bn)
    conv = prune(keras.layers.Conv2D(32, (3, 1), strides=1, padding='same'), decay(0.2, 0.4, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.4, 0.6, start_step, end_step))(conv)
    act = Mish()(bn)

    conv_1 = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(input)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(64, (1, 3), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(64, (3, 1), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(64, (1, 3), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = prune(keras.layers.Conv2D(32, (3, 1), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(act_1)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)

    mp = keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(input)
    conv_2 = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(mp)
    bn_2 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_2)
    act_2 = Mish()(bn_2)

    conv_3 = prune(keras.layers.Conv2D(32, (1, 1), strides=1, padding='same'), decay(0.3, 0.5, start_step, end_step))(input)
    bn_3 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.5, 0.7, start_step, end_step))(conv_3)
    act_3 = Mish()(bn_3)

    concat = keras.layers.concatenate(inputs=[act, act_1, act_2, act_3], axis=3)
    sum = keras.layers.add([concat, input])
    return sum

def neck_module2(ip, start_step, end_step, prune, decay):
    input = Mish()(ip)
    dws = prune(keras.layers.SeparableConv2D(64, (3, 3), padding='same'), decay(0.4, 0.7, start_step, end_step))(input)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.6, 0.8, start_step, end_step))(dws)
    act = Mish()(bn)
    dws = prune(keras.layers.SeparableConv2D(128, (3, 3), padding='same'), decay(0.4, 0.7, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.6, 0.8, start_step, end_step))(dws)
    act = Mish()(bn)
    dws = prune(keras.layers.SeparableConv2D(192, (3, 3), padding='same'), decay(0.4, 0.7, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.6, 0.8, start_step, end_step))(dws)
    act = Mish()(bn)
    dws_c = prune(keras.layers.SeparableConv2D(64, (3, 3), padding='same'), decay(0.4, 0.7, start_step, end_step))(input)
    bn_c = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.6, 0.8, start_step, end_step))(dws_c)
    act_c = Mish()(bn_c)
    concat = keras.layers.concatenate(inputs=[input, act_c], axis=3)
    sum = keras.layers.add([concat, act])
    act_sum = Mish()(sum)

    mp = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(act_sum)

    conv = prune(keras.layers.Conv2D(128, (1, 1), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(mp)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv)
    act = Mish()(bn)
    conv = prune(keras.layers.Conv2D(128, (1, 3), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv)
    act = Mish()(bn)
    conv = prune(keras.layers.Conv2D(256, (3, 1), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv)
    act = Mish()(bn)

    conv_01 = prune(keras.layers.Conv2D(96, (1, 3), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act)
    bn_01 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv_01)
    act_01 = Mish()(bn_01)
    conv_02 = prune(keras.layers.Conv2D(96, (3, 1), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act)
    bn_02 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv_02)
    act_02 = Mish()(bn_02)

    conv_1 = prune(keras.layers.Conv2D(192, (1, 1), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(mp)
    bn_1 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv_1)
    act_1 = Mish()(bn_1)

    conv_11 = prune(keras.layers.Conv2D(96, (1, 3), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act_1)
    bn_11 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv_11)
    act_11 = Mish()(bn_11)
    conv_12 = prune(keras.layers.Conv2D(96, (3, 1), strides=1, padding='same'), decay(0.6, 0.8, start_step, end_step))(act_1)
    bn_12 = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, start_step, end_step))(conv_12)
    act_12 = Mish()(bn_12)

    concat_1 = keras.layers.concatenate(inputs=[act_01, act_02], axis=3)
    concat_2 = keras.layers.concatenate(inputs=[act_11, act_12], axis=3)
    sum = keras.layers.add([mp, concat_1, concat_2])

    return sum

def head(ip, start_step, end_step, prune, decay):
    input = Mish()(ip)
    dws = prune(keras.layers.SeparableConv2D(256, (3, 3), padding='same'), decay(0.7, 0.9, 0, end_step))(input)
    bn_dws = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step))(dws)
    act_dws = Mish()(bn_dws)
    dws = prune(keras.layers.SeparableConv2D(384, (3, 3), padding='same'), decay(0.7, 0.9, 0, end_step))(act_dws)
    bn_dws = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step))(dws)
    act_dws = Mish()(bn_dws)

    conv = prune(keras.layers.Conv2D(384, (1, 1), strides=1, padding='same'), decay(0.7, 0.9, 0, end_step))(input)
    bn = prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step))(conv)
    act = Mish()(bn)

    sum = keras.layers.add([act, act_dws])
    act_sum = Mish()(sum)

    dws = prune(keras.layers.SeparableConv2D(512, (3, 3), padding='same'), decay(0.7, 0.9, 0, end_step))(act_sum)
    bn =  prune(keras.layers.BatchNormalization(axis=-1, momentum=0.9), decay(0.8, 0.9, 0, end_step))(dws)
    act = Mish()(bn)
    gap = keras.layers.GlobalAveragePooling2D()(act)
    return gap

def FADNet(input_shape, start_step, end_step, prune, decay):
    img_input = keras.layers.Input(shape=input_shape)
    bb = backbone(img_input, start_step, end_step, prune, decay)
    mp0 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(bb)
    n1 = neck_module1(mp0, start_step, end_step, prune, decay)
    mp1 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(n1)
    n2 = neck_module2(mp1, start_step, end_step, prune, decay)
    mp2 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(n2)
    h = head(mp2, start_step, end_step, prune, decay)
    d2 = prune(keras.layers.Dense(1), decay(0.7, 0.9, start_step, end_step))(h)
    d2 = Sigmoid()(d2)
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
        if val_acc >= self.val_threshold or tr_acc >= self.tr_threshold:
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

def blur(img):
    return (cv2.blur(img,(5,5)))

def main():
    # Start k-fold cross vailidation
    k = 6
    val_acc = 0
    val_loss = 0
    model = None
    select = 1

    # Uncomment this to visualize an example of the learn rate schedule
    # temp_learning_rate_schedule = CustomSchedule(initial_learning_rate=0.001*0.95**50, decay_steps=584, decay_rate=0.95, warmup_steps=0)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()

    fill_modes = ["constant", "reflect", "nearest"]
    rr = [10.0, 15.0, 10.0]
    sr = [10.0, 20.0, 10.0]
    cval = [np.random.random()*255, 0, 0]

    for i in range(3):
        # prefix = 'ks\\k' + str(i) + '\\'
        prefix = 'dataset/'
        # img_width, img_height = 100, 100
        train_data_dir = prefix + 'train/'
        validation_data_dir = prefix + 'test/'
        epochs = 30
        batch_size = 30

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
            # target_size=(img_width, img_height),
            target_size=(320, 320),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            # target_size=(img_width, img_height),
            target_size=(320, 320),
            batch_size=batch_size,
            class_mode='binary')

        # Get input image dimension
        unit_image = train_generator.next()[0]
        shape = (unit_image.shape[1], unit_image.shape[2], unit_image.shape[3])

        # pruning specifications
        poly_decay = tfmot.sparsity.keras.PolynomialDecay
        prune = tfmot.sparsity.keras.prune_low_magnitude
        start_step = 101
        end_step = int(584 * epochs / 2)
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        quantize_scope = tfmot.quantization.keras.quantize_scope

        # Pruned DNN
        pruned_model = FADNet(shape, start_step, end_step, prune, poly_decay)

        # BP
        decay_exp = 0
        if i > 0:
            decay_exp = 60
        # if i == 5:
        #     th = 0.985

        lr_schedule = CustomSchedule(initial_learning_rate=0.0001*(0.98**decay_exp)/(i+1),
                                    decay_steps=584,
                                    decay_rate=0.95,
                                    warmup_steps=584*int((i+1)/2))
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # check for transfer learning
        if os.path.exists("quantized_model.h5"):
            pruned_model.load_weights("quantized_model.h5", by_name=True)
            print(pruned_model.layers[-2].get_weights()[0])
            print("Transfer learning continued")

        pruned_model.summary()
        pruned_model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])

        # callbacks
        my_callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            # tfmot.sparsity.keras.PruningSummaries(log_dir='pruned_model'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            MyThresholdCallback(tr_threshold=0.99, val_threshold=0.992+0.001*i),
            # TensorBoard(log_dir="prune"+str(i),
            #             histogram_freq=0,
            #             write_graph=True,
            #             write_images=False,
            #             update_freq="epoch",
            #             profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='pruned_model.h5',
                                            save_weights_only=False,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
        ]
        history = pruned_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)

        # visualize compression
        # _, pruned_keras_file = tempfile.mkstemp('.h5')
        # tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        # print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

        print('Finished part ' + str(i + 1))

        # strip pruning layers for final model
        model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
        model_for_export.save("pruned_model_stripped.h5")
        model_for_export.save_weights("pruned_model_weights.h5")

if __name__ == '__main__':
    main()
