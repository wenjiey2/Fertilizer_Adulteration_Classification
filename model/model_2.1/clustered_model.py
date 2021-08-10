import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
import pruned_model as pm
import quantized_model as qm
import os
from tensorflow.python.ops import clustering_ops
import cluster_config
import six
import abc
import cluster as cc

k = tf.keras.backend
CentroidInitialization = cluster_config.CentroidInitialization
@six.add_metaclass(abc.ABCMeta)

class AbstractCentroidsInitialisation:
  """
  Abstract base class for implementing different cluster centroid
  initialisation algorithms. Must be initialised with a reference to the
  weights and implement the single method below.
  """

  def __init__(self, weights, number_of_clusters):
    self.weights = weights
    self.number_of_clusters = number_of_clusters

  @abc.abstractmethod
  def get_cluster_centroids(self):
    pass

# class NonZeroDensityBasedCentroidsInitialisation(AbstractCentroidsInitialisation):
#   """
#   This initialisation means that we build a cumulative distribution
#   function(CDF), then linearly space y-axis of this function then find the
#   corresponding x-axis points. In order to simplify the implementation, here is
#   a plan how it is achieved:
#   1. Calculate CDF values at points spaced linearly between weight_min and
#   weight_max(e.g. 20 points)
#   2. Build an array of values linearly spaced between 0 and 1(probability)
#   3. Go through the second array and find segment of CDF that contains this
#   y-axis value, \\hat{y}
#   4. interpolate linearly between those two points, get a line equation y=ax+b
#   5. solve equation \\hat{y}=ax+b for x. The found x value is a new cluster
#   centroid
#   """
#
#   def get_cluster_centroids(self):
#     mask = tf.math.equal(self.weights, 0)
#     mask = tf.math.logical_not(mask)
#     non_zero_weights = tf.boolean_mask(self.weights, mask)
#     weight_min = tf.reduce_min(non_zero_weights)
#     weight_max = tf.reduce_max(non_zero_weights)
#     # Calculating interpolation nodes, +/- 0.01 is introduced to guarantee that
#     # CDF will have 0 and 1 and the first and last value respectively.
#     # The value 30 is a guess. We just need a sufficiently large number here
#     # since we are going to interpolate values linearly anyway and the initial
#     # guess will drift away. For these reasons we do not really
#     # care about the granularity of the lookup.
#     cdf_x_grid = tf.linspace(weight_min - 0.01, weight_max + 0.01, 30)
#
#     f = TFCumulativeDistributionFunction(weights=non_zero_weights)
#
#     cdf_values = k.map_fn(f.get_cdf_value, cdf_x_grid)
#
#     probability_space = tf.linspace(0 + 0.01, 1, self.number_of_clusters - 1)
#
#     # Use upper-bound algorithm to find the appropriate bounds
#     matching_indices = tf.searchsorted(sorted_sequence=cdf_values,
#                                        values=probability_space,
#                                        side='right')
#
#     # Interpolate linearly between every found indices I at position using I at
#     # pos n-1 as a second point. The value of x is a new cluster centroid
#     def get_single_centroid(i):
#       i_clipped = tf.minimum(i, tf.size(cdf_values) - 1)
#       i_previous = tf.maximum(0, i_clipped - 1)
#
#       s = TFLinearEquationSolver(x1=cdf_x_grid[i_clipped],
#                                  y1=cdf_values[i_clipped],
#                                  x2=cdf_x_grid[i_previous],
#                                  y2=cdf_values[i_previous])
#
#       y = cdf_values[i_clipped]
#
#       single_centroid = s.solve_for_x(y)
#       return single_centroid
#
#     centroids = k.map_fn(get_single_centroid,
#                          matching_indices,
#                          dtype=tf.float32)
#     print(centroids)
#     centroids = tf.concat(0.0, centroids, 0)
#     print(centroids)
#     cluster_centroids = tf.reshape(centroids, (self.number_of_clusters,))
#     return cluster_centroids
#
# class TFLinearEquationSolver:
#   """
#   Solves a linear equantion y=ax+b for either y or x.
#   The line equation is defined with two points (x1, y1) and (x2,y2)
#   """
#
#   def __init__(self, x1, y1, x2, y2):
#     self.x1 = x1
#     self.y1 = y1
#     self.x2 = x2
#     self.y2 = y2
#
#     # Writing params for y=ax+b
#     self.a = (y2 - y1) / tf.maximum(x2 - x1, 0.001)
#     self.b = y1 - x1 * ((y2 - y1) / tf.maximum(x2 - x1, 0.001))
#
#   def solve_for_x(self, y):
#     """
#     For a given y value, find x at which linear function takes value y
#     :param y: the y value
#     :return: the corresponding x value
#     """
#     return (y - self.b) / self.a
#
#   def solve_for_y(self, x):
#     """
#     For a given x value, find y at which linear function takes value x
#     :param x: the x value
#     :return: the corresponding y value
#     """
#     return self.a * x + self.b
#
#
# class TFCumulativeDistributionFunction:
#   """
#   Takes an array and builds cumulative distribution function(CDF)
#   """
#
#   def __init__(self, weights):
#     self.weights = weights
#
#   def get_cdf_value(self, given_weight):
#     mask = tf.less_equal(self.weights, given_weight)
#     less_than = tf.cast(tf.math.count_nonzero(mask), dtype=tf.float32)
#     return less_than / tf.size(self.weights, out_type=tf.float32)

class NonZeroKmeansPlusPlusCentroidsInitialisation(AbstractCentroidsInitialisation):
  """
  Cluster centroids based on kmeans++ algorithm
  """
  def get_cluster_centroids(self):

    weights = tf.reshape(self.weights, [-1, 1])

    cluster_centroids = clustering_ops.kmeans_plus_plus_initialization(weights,
                                                                       self.number_of_clusters,
                                                                       seed=9,
                                                                       num_retries_per_sample=-1)

    return cluster_centroids

class CentroidsInitializerFactory:
  """
  Factory that creates concrete initializers for factory centroids.
  To implement a custom one, inherit from AbstractCentroidsInitialisation
  and implement all the required methods.
  After this, update CentroidsInitialiserFactory.__initialisers hashtable to
  reflect new methods available.
  """
  _initialisers = {
      # CentroidInitialization.DENSITY_BASED: NonZeroDensityBasedCentroidsInitialisation
      CentroidInitialization.KMEANS_PLUS_PLUS: NonZeroKmeansPlusPlusCentroidsInitialisation
  }

  @classmethod
  def init_is_supported(cls, init_method):
    return init_method in cls._initialisers

  @classmethod
  def get_centroid_initializer(cls, init_method):
    """
    :param init_method: a CentroidInitialization value representing the init
      method requested
    :return: A concrete implementation of AbstractCentroidsInitialisation
    :raises: ValueError if the requested centroid initialization method is not
      recognised
    """
    return cls._initialisers[init_method]

# custom cluster layers
def apply_clustering(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        if layer.name == 'conv2d_23':
            return cc.cluster_weights(layer, 256, CentroidInitialization.KMEANS_PLUS_PLUS)
        elif layer.output_shape[-2] == 5:
            return cc.cluster_weights(layer, 128, CentroidInitialization.KMEANS_PLUS_PLUS)
    elif isinstance(layer, tf.keras.layers.Dense):
        return cc.cluster_weights(layer, 32, CentroidInitialization.KMEANS_PLUS_PLUS)
    return layer

def main():
    # data preperation
    k = 6
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
        train_data_dir = prefix + 'train/'
        validation_data_dir = prefix + 'test/'
        epochs = 20
        batch_size = 30

        if select == 2:
            select = 0
        else:
            select += 1
        print(fill_modes[select])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.1,
            height_shift_range=0.1,
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

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size = (320, 320),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size = (320, 320),
            batch_size=batch_size,
            class_mode='binary')

        lr_schedule = pm.CustomSchedule(initial_learning_rate=0.0001/(i+1), decay_steps=584, decay_rate=0.9, warmup_steps=584*2)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1)
        bcel = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        my_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        pm.MyThresholdCallback(tr_threshold=0.995, val_threshold=0.98+0.005*i),
                        keras.callbacks.ModelCheckpoint(filepath='clustered_model.h5',
                                                        save_weights_only=False,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True)
                        # keras.callbacks.TensorBoard(log_dir="stats\\cluster_fold"+str(i),
                        #                             histogram_freq=0,
                        #                             write_graph=True,
                        #                             write_images=True,
                        #                             update_freq="epoch",
                        #                             profile_batch=2)
                        ]

        # load quantized model
        if not os.path.exists("clustered_model.h5"):
            unit_image = train_generator.next()[0]
            shape = (unit_image.shape[1], unit_image.shape[2], unit_image.shape[3])
            poly_decay = tfmot.sparsity.keras.PolynomialDecay
            prune = tfmot.sparsity.keras.prune_low_magnitude
            end_step = 584 * epochs
            start_step = 101
            base_model = pm.FADNet(shape, start_step, end_step, prune, poly_decay)
            base_model = tfmot.sparsity.keras.strip_pruning(base_model)
            base_model.load_weights("pruned_model_2_weights.h5", by_name=True)
            #print(base_model.layers[-2].get_weights()[0])

            # weight clustering
            # base_model.summary()
            clustered_model = tf.keras.models.clone_model(base_model, clone_function=apply_clustering)
            clustered_model.summary()

        print(tfmot.clustering.keras.strip_clustering(clustered_model).layers[-2].get_weights()[0])
        clustered_model.compile(optimizer=opt, loss=bcel, metrics=['accuracy'])

        # fine-tune
        clustered_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=my_callbacks)
        clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)
        clustered_model.save('clustered_model.h5')

    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(clustered_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open("clustered_model_fl32.tflite", "wb").write(tflite_model)

    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open("clustered_model_int8.tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    main()
