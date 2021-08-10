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
import pruned_model as pm
import tensorflow_addons as tfa

def main():
    # Start k-fold cross vailidation
    k = 6
    val_acc = 0
    val_loss = 0
    model = None
    select = 0

    # Uncomment this to visualize an example of the learn rate schedule
    # temp_learning_rate_schedule = CustomSchedule(initial_learning_rate=0.001*0.95**50, decay_steps=584, decay_rate=0.95, warmup_steps=0)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()

    epochs = 20
    batch_size = 20

    # pruning specifications
    poly_decay = tfmot.sparsity.keras.PolynomialDecay
    prune = tfmot.sparsity.keras.prune_low_magnitude
    start_step = 21
    end_step = 950
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
    quantize_scope = tfmot.quantization.keras.quantize_scope

    prefix = 'fert_dataset/'
    train_data_dir = prefix + 'train/'
    validation_data_dir = prefix + 'test/'

    # Generate Tensor for input images
    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.7, 1.2),
            zoom_range=(0.8, 1.3),
            channel_shift_range=60,
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

    # check for transfer learning
    pruned_fnf_model = pm.FADNet(shape, start_step, end_step, prune, poly_decay)
    ilr = 0.001
    if os.path.exists("quantized_fnf_model.h5") and os.path.exists("pruned_fnf_model_2_stripped.h5"):
        ilr = 0.0001
        pruned_fnf_model.load_weights("pruned_fnf_model_2_stripped.h5", by_name=True)
        print("Transfer learning continued")
    # print(pruned_fnf_model.layers[-2].get_weights()[0])
    pruned_fnf_model.summary()

    # BP
    lr_schedule = pm.CustomSchedule(initial_learning_rate=ilr,
                                    decay_steps=80,
                                    decay_rate=0.95,
                                    warmup_steps=21)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1)
    fl = tfa.losses.SigmoidFocalCrossEntropy()

    pruned_fnf_model.compile(optimizer=opt, loss=fl, metrics=['accuracy'])

    # callbacks
    my_callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            # tfmot.sparsity.keras.PruningSummaries(log_dir='pruned_fnf_model'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            pm.MyThresholdCallback(tr_threshold=0.985, val_threshold=0.998),
            # TensorBoard(log_dir="prune"+str(i),
            #             histogram_freq=0,
            #             write_graph=True,
            #             write_images=False,
            #             update_freq="epoch",
            #             profile_batch=2),
            keras.callbacks.ModelCheckpoint(filepath='pruned_fnf_model_2.h5',
                                            save_weights_only=False,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
    ]
    history = pruned_fnf_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)

        # visualize compression
        # _, pruned_keras_file = tempfile.mkstemp('.h5')
        # tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        # print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

        # strip pruning layers for final model
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_fnf_model)
    model_for_export.save("pruned_fnf_model_2_stripped.h5")
    model_for_export.save_weights("pruned_fnf_model_2_weights.h5")

if __name__ == '__main__':
    main()
