import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import os
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
#tf.config.list_physical_devices('GPU')

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

# mish Activation
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

# Helper function that loads images
def load(file):
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# predict on subimages
def sub_predict(prefix, model):
    adulterates = sorted(glob.glob(prefix + 'test\\adulterated\\*.jpg'))
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

    cleans = sorted(glob.glob(prefix + 'test\\clean\\*.jpg'))
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

# customize DNN
def my_DNN(input_shape):
    img_input = keras.layers.Input(shape=input_shape)
    conv_0 = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation=mish)(img_input)
    bn_0 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_0)

    dws_1 = keras.layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False, activation=mish)(bn_0)
    bn_dws_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(dws_1)
    res_1 = keras.layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', activation=mish)(bn_0)
    bn_res_1 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(res_1)

    conv_11 = tf.keras.layers.Conv2D(32, (1, 1), strides=1, padding='same', activation=mish)(bn_dws_1)
    bn_11 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_11)
    conv_12 = keras.layers.Conv2D(32, (1, 3), strides=1, padding='same', activation=mish)(bn_11)
    bn_12 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_12)
    conv_13 = keras.layers.Conv2D(32, (3, 1), strides=1, padding='same', activation=mish)(bn_12)
    bn_13 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_13)
    conv_14 = keras.layers.Conv2D(32, (1, 1), strides=1, padding='same', activation=mish)(bn_13)
    bn_14 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_14)
    mp_1 = keras.layers.MaxPooling2D((2, 2))(bn_14)
    mp_1 = keras.layers.add([mp_1, bn_res_1])

    conv_21 = keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation=mish)(mp_1)
    bn_21 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_21)
    conv_22 = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same', activation=mish)(bn_21)
    bn_22 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_22)
    conv_23 = keras.layers.Conv2D(64, (3, 1), strides=1, padding='same', activation=mish)(bn_22)
    bn_23 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_23)
    conv_24 = keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation=mish)(bn_23)
    bn_24 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_24)

    conv_31 = keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation=mish)(mp_1)
    bn_31 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_31)
    conv_32 = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same', activation=mish)(bn_31)
    bn_32 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_32)
    conv_33 = keras.layers.Conv2D(64, (3, 1), strides=1, padding='same', activation=mish)(bn_32)
    bn_33 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_33)
    conv_34 = keras.layers.Conv2D(64, (1, 3), strides=1, padding='same', activation=mish)(bn_33)
    bn_34 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_34)
    conv_35 = keras.layers.Conv2D(64, (3, 1), strides=1, padding='same', activation=mish)(bn_34)
    bn_35 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_35)

    mp_2 = keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(mp_1)
    conv_41 = keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation=mish)(mp_2)
    bn_41 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_41)

    conv_51 = keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation=mish)(mp_1)
    bn_51 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_51)

    concat_1 = tf.keras.layers.concatenate(inputs=[bn_24, bn_35, bn_41, bn_51], axis=3)

    mp_5 = keras.layers.MaxPooling2D((2, 2))(concat_1)

    conv_71 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=mish)(mp_5)
    bn_71 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_71)
    conv_72 = keras.layers.Conv2D(128, (1, 3), strides=1, padding='same', activation=mish)(bn_71)
    bn_72 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_72)
    conv_73 = keras.layers.Conv2D(128, (3, 1), strides=1, padding='same', activation=mish)(bn_71)
    bn_73 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_73)
    conv_74 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=mish)(mp_5)
    bn_74 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_74)
    conv_75 = keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation=mish)(bn_74)
    bn_75 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_75)
    conv_76 = keras.layers.Conv2D(128, (3, 1), strides=1, padding='same', activation=mish)(bn_75)
    bn_76 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_76)
    conv_77 = keras.layers.Conv2D(128, (1, 3), strides=1, padding='same', activation=mish)(bn_75)
    bn_77 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_77)
    conv_78 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=mish)(mp_5)
    bn_78 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_78)
    concat_2 = tf.keras.layers.concatenate(inputs=[bn_76, bn_77, bn_72, bn_73, bn_78], axis=3)

    dws_2 = keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, activation=mish)(concat_2)
    bn_dws_2 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(dws_2)
    res_2 = keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', activation=mish)(concat_2)
    bn_res_2 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(res_2)

    conv_81 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=mish)(bn_dws_2)
    bn_81 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_81)
    conv_82 = keras.layers.Conv2D(128, (1, 3), strides=1, padding='same', activation=mish)(bn_81)
    bn_82 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_82)
    conv_83 = keras.layers.Conv2D(128, (3, 1), strides=1, padding='same', activation=mish)(bn_82)
    bn_83 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_83)
    conv_84 = keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation=mish)(bn_83)
    bn_84 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv_84)

    mp_6 = keras.layers.MaxPooling2D((2, 2))(bn_84)
    mp_6 = keras.layers.add([mp_6, bn_res_2])

    dws_3 = keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, activation=mish)(mp_6)
    bn_dws_3 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(dws_3)
    dws_4 = keras.layers.SeparableConv2D(512, (3, 3), padding='same', use_bias=False, activation=mish)(bn_dws_3)
    bn_dws_4 = keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(dws_4)

    gap = keras.layers.GlobalAveragePooling2D()(bn_dws_4)

    fl = keras.layers.Flatten()(gap)
    d2 = keras.layers.Dense(1, activation='sigmoid')(fl)
    model = tf.keras.Model(img_input, d2)
    return model

def main():
    # Start k-fold cross vailidation
    k = 6
    predictions = {}
    val_acc = 0
    val_loss = 0
    model = None
    select = -1

    # plot of an example learn rate schedule
    # temp_learning_rate_schedule = CustomSchedule(initial_learning_rate=0.001*0.95**50, decay_steps=870, decay_rate=0.95, warmup_steps=0)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()

    fill_modes = ["constant", "reflect"]

    for i in range(k):
        prefix = 'ks\\k' + str(i) + '\\'
        img_width, img_height = 100, 100
        train_data_dir = prefix + 'train\\'
        validation_data_dir = prefix + 'test\\'
        epochs = 50 + 10*i
        batch_size = 30

        if select == -1:
            select = np.random.randint(0, 2)
        elif select == 0:
            select = 1
        elif select:
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
        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

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

        # DNN model
        model = my_DNN(shape)

        model.summary()

        # BP
        th = 0.97
        decay_exp = 0
        if i > 0:
            decay_exp = epochs - 10
            th = 0.999
        if i == 5:
            th = 0.98

        lr_schedule = CustomSchedule(initial_learning_rate=0.0005*(0.95**decay_exp)/(i+1), decay_steps=522, decay_rate=0.95, warmup_steps=5220*int((i+1)/2))
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        if os.path.exists("new_model.h5"):
            model = keras.models.load_model("new_model.h5", custom_objects={"mish":mish}, compile=False)
            print("Transfer learning continued")
        model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])
        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20+5*i, restore_best_weights = True),
            MyThresholdCallback(tr_threshold=0.94, val_threshold=th),
            keras.callbacks.TensorBoard(log_dir="stats\\fold"+str(i), histogram_freq=0, write_graph=True, write_images=False, update_freq="epoch", profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='new_model.h5', save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
        ]
        history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)
        print("Model update")

        print('Finished part ' + str(i + 1))

    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("new_model.tflite", "wb").write(tflite_model)

    # generate prediction counts
    for i in range(k):
        sub_predict('ks\\k' + str(i) + '\\', model)

if __name__ == '__main__':
    main()
