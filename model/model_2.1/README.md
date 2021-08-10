# Model 2.1

### Existing features of Model 2.0
- [x] stacking Conv2D & FC layers of ascending sizes (Basic idea from AlexNet)
- [x] Adam optimizer
- [x] Batch Norm
- [x] Exponential decay learning rate schedule
- [x] Customize callback for early stopping

### To-Do List of Model 2.1:
- [x] Mish Activation
- [x] Label Smoothing for Binary Cross Entropy loss
- [x] Gradient Clipping
- [x] Depth-wise Separable Convolution
- [x] Replace FC layers with Global Average Pooling
- [x] Inception-v3 Module A
- [x] Inception-v3 Module B
- [x] Residual connections
- [x] Data augmentation -- shift, flip, rotate, shear, channel-shift, brightness range, zoom range
- [x] Data augmentation -- alternating fill modes: reflect & constant
- [x] Transfer learning with K-folds on randomized fill mode
- [x] Fold-dependent, serialized learning rate schedule -- warm up & exponential decay
- [x] Tensorboard visualization
- [x] Fold-dependent early stopping
- [ ] Weight Pruning
- [x] Post-training quantization
- [ ] Quantization-aware training
- [ ] Focal loss
- [x] YOLOv4 region of interest detection -- customized training & inference (darknet)
- [x] YOLOv4 region of interest detection -- inference (tensorflow)
- [x] YOLOv4 post-training quantization for mobile device (tensorflow)
- [x] Cropping effectiveness analysis

### Major updates of model 2.1 from model 2.0:
1. Dataset is better augmented.
2. Increased DNN complexity with cascade of submodules.
3. Parameter reduction from ~20M to ~1M with minimal loss in complexity for light-weight deployment on edge device. Techniques including stacking 1x1, 1x3, 3x1 layers presented in Inception-v3, and depth-wise & point-wise convolutions to reduce both computational & representational cost.
3. Applied transfer learning with k-fold cross validation such that only one model is created and trained accumulatively instead of independently and repeatedly.
4. More advanced lr schedule and early stopping conditions.
5. Different quantization options for the trained model.
5. The stats folder contains our DNN architecture (FadNet.png), the plots of training accuracy & loss of all folds (with smoothing = 0.6), and the final accuracy vs. threshold plot for three different quantization levels.
6. YOLOv4 region of interest detection for image cropping.

### Training with your own images
You can train your own model on your PC for prediction if that is what's desired. With the current model complexity, a GPU is strongly recommended to reduce the training time.

#### dataset preparation
Images are resized and partitioned evenly into 12 equally-sized subimages: 4 (horizontally) by 3 (vertically), where the DNN training & inference are being done on these subimages. The reason is that while adulterants are pretty evenly distributed when we prepare the dataset, in reality, it may not be. In order to detect adulterants in any areas of the image taken in real life, and then make a comprehensive decision on whether the entire image is adulterated, we have set our standard for pure detection to be 8 pure subimages.

- preprocess.py - preprocesses images: (a) rotating the image if necessary (b) resizing the image to 400 * 300 (c) slicing the image into 12 100 * 100 pieces. To use this file, you need to either put your images into one of these five folders: *clumped*, *clumped_maize*, *discolored*, *maize*, *normal* under both *clean* and *adulterated* folder, or modified the directories in this file accordingly.
- partition.py - copies preprocessed images into multiple folders for a k-fold evaluation. In this case the folder for each partition is called "*ks\k[numer of partition]*", but you can name it something else at least it is consistent with the py file. To use this file, you may want to change *k* and file directories. *k* is recommended to be set between 5 and 10.

#### Training & Validation
- new_model.py - saves the DNN architecture trained through k-fold cross validation in new_model.h5 and new_model.tflite for andriod app. The model would be used for classification on partitioned subimages. The true predictions (out of 12 subimages) would be counted for each original image, and stored in a dictionary format of "*{image_name: pure_counts}*" in predictions.csv, which would be used in evaluate.py. Changes to this file are strongly not recommended.
- evaluate.py - evaluates accuracy for each of the five types of fertilizer for all thresholds (0-12). It first reads in a csv file (in the format of "[*image filename*], [*number of pieces classified as clean*]"), which contains how each image was predicted. It then reads in image files to record the real labels. Then for each threshold it forms a confusion matrix, then plots the threshold vs. accuracy rate for each type of image, and overall accuracy rate. Notice that the best threshold this file returns should only be used as a reference, especially when you have limited training images.

#### Testing on a new dataset
- test.py - predicts on a set of field images and prints out the number and percentage of images correctly classified. In our test case, we have prepared photos of pure fertilizers poorly taken with a large portion of background to test the model's robustness and the effectiveness of cropping. If you want to use the file on cropped full images, you may need to line 23-24 and disable line 27-63 for partition first, and set *dir* accordingly with flag *--cropped=True*. Notice that our demo code only counts the correct labels for pure images. You may need to the code slightly for counting correct adulterated predictions.

### Inference with different precisions
- quantize.py - quantize new_model.h5 model to a .tflite model with customized precision level of the input flag *--precision=* (one of fl32, fl16, int8). The .tflite model generated by new_model.py uses the standard 32-bit floating-point precision.

**The inference model can be found [here](https://drive.google.com/drive/folders/1UmpKOp49h79Y2rPL2AdQm6a2ht6zzffG?usp=sharing).**

If accuracy is being prioritized without any restraints on your computational resources, please run inference with .h5 file. Meanwhile, we recommend using 8-bit fixed-point quantization for inference on an edge device because it can achieve upto 4x storage reduction & speedup in inference while having minimal loss in accuracy. Notice the half-precision float16 is currently not supported by Android devices.

### References
- Batch Norm: https://arxiv.org/pdf/1502.03167.pdf
- LR Warmup: https://arxiv.org/pdf/1810.13243.pdf
- Mish Activation: https://arxiv.org/pdf/1908.08681.pdf
- Global Average Pooling: https://arxiv.org/pdf/1312.4400.pdf
- ResNet: https://arxiv.org/pdf/1512.03385.pdf
- Inception v3: https://arxiv.org/pdf/1512.00567.pdf
- Inception v4: https://arxiv.org/pdf/1602.07261.pdf
- Mobile Net: https://arxiv.org/pdf/1704.04861.pdf
- Focal Loss: https://arxiv.org/pdf/1708.02002.pdf

Lastly, if you have any questions, please contact Wenjie Yu.
