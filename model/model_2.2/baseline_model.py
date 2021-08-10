import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from tensorflow.keras.callbacks import TensorBoard
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
def backbone(input):
    conv = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    res = keras.layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same')(act)
    bn_res = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(res)
    act_res = Mish()(bn_res)

    dws = keras.layers.SeparableConv2D(16, (3, 3), padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)
    dws = keras.layers.SeparableConv2D(16, (3, 3), padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)

    mp = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(act)
    mp = keras.layers.add([mp, act_res])
    mp = Mish()(mp)

    conv = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(mp)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)
    conv = keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')(bn)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    conv_1 = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(mp)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(32, (5, 1), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(32, (1, 5), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)

    concat = tf.keras.layers.concatenate(inputs=[act, act_1], axis=3)
    return concat

def neck_module1(ip):
    input = Mish()(ip)
    conv = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)
    conv = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)
    conv = keras.layers.Conv2D(32, (3, 1), strides=1, padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    conv_1 = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(input)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(64, (3, 1), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)
    conv_1 = keras.layers.Conv2D(32, (3, 1), strides=1, padding='same')(act_1)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)

    mp = keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(input)
    conv_2 = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(mp)
    bn_2 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_2)
    act_2 = Mish()(bn_2)

    conv_3 = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')(input)
    bn_3 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_3)
    act_3 = Mish()(bn_3)

    concat = keras.layers.concatenate(inputs=[act, act_1, act_2, act_3], axis=3)
    sum = keras.layers.add([concat, input])
    return sum

def neck_module2(ip):
    input = Mish()(ip)
    dws = keras.layers.SeparableConv2D(64, (3, 3), padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)
    dws = keras.layers.SeparableConv2D(128, (3, 3), padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)
    dws = keras.layers.SeparableConv2D(192, (3, 3), padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)
    dws_c = keras.layers.SeparableConv2D(64, (3, 3), padding='same')(input)
    bn_c = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws_c)
    act_c = Mish()(bn_c)
    concat = keras.layers.concatenate(inputs=[input, act_c], axis=3)
    sum = keras.layers.add([concat, act])
    act_sum = Mish()(sum)

    mp = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(act_sum)

    conv = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same')(mp)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)
    conv = keras.layers.Conv2D(128, (1, 3), strides=1, padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)
    conv = keras.layers.Conv2D(256, (3, 1), strides=1, padding='same')(act)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    conv_01 = keras.layers.Conv2D(96, (1, 3), strides=1, padding='same')(act)
    bn_01 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_01)
    act_01 = Mish()(bn_01)
    conv_02 = keras.layers.Conv2D(96, (3, 1), strides=1, padding='same')(act)
    bn_02 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_02)
    act_02 = Mish()(bn_02)

    conv_1 = keras.layers.Conv2D(192, (1, 1), strides=1, padding='same')(mp)
    bn_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_1)
    act_1 = Mish()(bn_1)

    conv_11 = keras.layers.Conv2D(96, (1, 3), strides=1, padding='same')(act_1)
    bn_11 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_11)
    act_11 = Mish()(bn_11)
    conv_12 = keras.layers.Conv2D(96, (3, 1), strides=1, padding='same')(act_1)
    bn_12 = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv_12)
    act_12 = Mish()(bn_12)

    concat_1 = keras.layers.concatenate(inputs=[act_01, act_02], axis=3)
    concat_2 = keras.layers.concatenate(inputs=[act_11, act_12], axis=3)
    sum = keras.layers.add([mp, concat_1, concat_2])

    return sum

def head(ip):
    input = Mish()(ip)
    dws = keras.layers.SeparableConv2D(256, (3, 3), padding='same')(input)
    bn_dws = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act_dws = Mish()(bn_dws)
    dws = keras.layers.SeparableConv2D(384, (3, 3), padding='same')(act_dws)
    bn_dws = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act_dws = Mish()(bn_dws)

    conv = keras.layers.Conv2D(384, (1, 1), strides=1, padding='same')(input)
    bn = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(conv)
    act = Mish()(bn)

    sum = keras.layers.add([act, act_dws])
    act_sum = Mish()(sum)

    dws = keras.layers.SeparableConv2D(512, (3, 3), padding='same')(act_sum)
    bn =  keras.layers.BatchNormalization(axis=-1, momentum=0.9)(dws)
    act = Mish()(bn)
    gap = keras.layers.GlobalAveragePooling2D()(act)
    return gap

def FADNet(input_shape):
    img_input = keras.layers.Input(shape=input_shape)
    bb = backbone(img_input)
    mp0 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(bb)
    n1 = neck_module1(mp0)
    mp1 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(n1)
    n2 = neck_module2(mp1)
    mp2 = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(n2)
    h = head(mp2)
    d2 = keras.layers.Dense(1)(h)
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
    val_acc = 0
    val_loss = 0
    model = None
    select = 1

    # Uncomment this to visualize an example of the learn rate schedule
    # temp_learning_rate_schedule = CustomSchedule(initial_learning_rate=0.001*0.95**50_steps=522_rate=0.95, warmup_steps=0)
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
        epochs = 20
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
                                          horizontal_flip=True,
                                          vertical_flip=True)

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

        bl_model = FADNet(shape)

        # BP
        lr_schedule = CustomSchedule(initial_learning_rate=0.0008*(0.98**(epochs*i/2))/(i+1),
                                    decay_steps=584,
                                    decay_rate=0.95,
                                    warmup_steps=584*int((i+1)/2))
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=0.0001, clipnorm=1)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # check for transfer learning
        if os.path.exists("bl_model.h5"):
            bl_model = keras.models.load_model("bl_model.h5",
                                               custom_objects={'Mish': Mish, 'Sigmoid': Sigmoid},
                                               compile=False)
            #bl_model = tf.keras.models.clone_model(base_model, clone_function=apply_pruning_to_custom_layers)
            print("Transfer learning continued")

        bl_model.summary()
        bl_model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])

        # callbacks
        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5+i, restore_best_weights=True),
            MyThresholdCallback(tr_threshold=0.965+0.01*i, val_threshold=0.975+0.01*i),
            # TensorBoard(log_dir="prune"+str(i),
            #             histogram_freq=0,
            #             write_graph=True,
            #             write_images=False,
            #             update_freq="epoch",
            #             profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='bl_model.h5',
                                            save_weights_only=False,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
        ]
        history = bl_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)

        # visualize compression
        # _, pruned_keras_file = tempfile.mkstemp('.h5')
        # tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        # print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

        print('Finished part ' + str(i + 1))

if __name__ == '__main__':
    main()
