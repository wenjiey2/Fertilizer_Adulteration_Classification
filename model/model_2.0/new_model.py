import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from matplotlib.pyplot import MultipleLocator
# print(len(glob.glob('ks/k1/train/adulterated/*.jpg')))
# print(len(glob.glob('ks/k1/train/clean/*.jpg')))

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

def load(file):
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def save_stats(model, history):
    model.save('new_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("new_model.tflite", "wb").write(tflite_model)

    plt.figure()
    plt.plot(history.history['loss'], label='tr_loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim([0, 1])
    plt.legend(loc='top right')
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(history.history['accuracy'], label='tr_acc')
    plt.plot(history.history['val_accuracy'], label = 'val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.ylim([0, 1])
    plt.legend(loc='lower right')
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.savefig('accuracy.png')

def sub_predict(prefix):
    adulterates = sorted(glob.glob(prefix + 'test/adulterated/*.jpg'))
    pics = int(len(adulterates) / 12)
    for p in range(pics):
        cropped = adulterates[p * 12: (p + 1) * 12]
        acc = 0
        for piece in cropped:
            prediction = model.predict(load(piece))[0][0]
            if prediction > 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('/') + 1: filename.rfind('_')]
        predictions[filename] = acc

    cleans = sorted(glob.glob(prefix + 'test/clean/*.jpg'))
    pics = int(len(cleans) / 12)
    for p in range(pics):
        cropped = cleans[p * 12: (p + 1) * 12]
        acc = 0
        for piece in cropped:
            prediction = model.predict(load(piece))[0][0]
            if prediction > 0.5:
                acc += 1
        filename = cropped[0]
        filename = filename[filename.rfind('/') + 1: filename.rfind('_')]
        predictions[filename] = acc

    # save pure prediction counts of subimages for test images
    with open('predictions.csv', 'w') as f:
        for key in predictions.keys():
            f.write("%s,%s\n" % (key, predictions[key]))

k = 6
predictions = {}
val_acc = 0
val_loss = 0

# K-fold cross vailidation
for i in range(k):
    prefix = 'ks/k' + str(i) + '/'
    img_width, img_height = 100, 100
    train_data_dir = prefix + 'train/'
    validation_data_dir = prefix + 'test/'
    epochs = 20
    batch_size = 15

    # Generate Tensor for input images
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
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

    # DNN model
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu', input_shape=shape),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    # BP
    decay_steps = 1566*12*(k-1)/k/batch_size*epochs
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=decay_steps,
        decay_rate=0.1)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights = True),
        MyThresholdCallback(tr_threshold=0.91, val_threshold=0.94)
    ]
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)

    # save model & stats
    test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
    if test_acc > val_acc:
        val_acc = test_acc
        val_loss = test_loss
        save_stats(model, history)
    elif test_acc == val_acc:
        if test_loss < val_loss:
            val_acc = test_acc
            val_loss = test_loss
            save_stats(model, history)

    # predict on 12 subimages of test images
    sub_predict(prefix)

    print('Finished part ' + str(i + 1))
#print(predictions)
