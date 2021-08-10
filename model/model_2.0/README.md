- evaluate.py - evaluates accuracy rates of each type of image, including clumped, clumped with maize, discolored, with maize and normal at different thresholds. It first reads in a csv file (int the format of "[*image filename*], [*number of pieces classified as clean*]"), which contains how each image was predicted. It then reads in image files to record the real labels. Then for each threshold it forms a confusion matrix, then plots the threshold vs. accuracy rate for each type of image, and overall accuracy rate.
- test.py - predict on field images of pure fertilizers and prints out the number and percentage of images correctly classified.
- new_model.py - save the DNN architecture trained through k-fold validation in new_model.h5 and new_model.tflite for andriod app. The model would be used for classification for partitioned subimages. The  true predictions (out of 12 subimages) would be counted for each original image, and stored in a dictionary format of "*{image_name: pure_counts}*" in predictions.csv, which would be used in evaluate.py.
- partition.py - copys preprocessed images into multiple folders for a k-fold evaluation. In this case the folder for each partition is called "*ks\k[numer of partition]*", but you can name it something else at least it is consistent with the py file. To use this file, you need to change your desired *k* and file directories.
- preprocess.py - preprocesses and slice images, which includes: (a) rotating the image if necessary (b) resizing the image to 400 * 300 (c) slicing the image into 12 100 * 100 pieces. To use this file, you probably need to rewrite the file directories, but the preprocess function is already defined.

This is where the model will be updated starting from Summer 2020, since there may be major changes to our DNN training & inferences procedure in the future. The currently version aimed to preserve most files and the core idea in Eric Wang's model, while accomendating for the updated Tensorflow & Keras API and the directory format of MacOS.

The major changes are as follows:
1. Merged knn.py and model.py (or got rid of model.py if you like) into new_model.py to generate a trained DNN stored in new_model.h5 and new_model.tflite for our android app. The previous model.h5 file generated was not the result of k-fold cross validation. Therefore, the model's optimization at that stage cannot be guaranteed. In fact, since we can observe a noticeable amount of variation in the training and validation accuracy for different folds, this could be the factor the obvious false predictions in testing.
2. Avoided cases where we potentially validate over the training set.
3. Fixed data truncation error with *k* in k-fold set to 6 (a divisible number by the total dataset size) to save training time.
4. Stacked more Conv2D layers with larger feature maps in DNN architecture following the classical AlexNet structure. In order to compensate for the lack of computation resources, our network is still comparably smaller and shallower than most of the famous networks out there.
5. Basic tuning techniques were used. Batch norm layers boost the initial accuracy by a great amount. Early stop is applied to each fold where training will stop once validation accuracy reaches 92%, or does not increase for three consecutive epochs to avoid overfitting. Descending learning rate schedule is applied to maximize learning effectiveness in later epochs. Relatively large fluctuating validation accuracy still exists (for the current version), but the issue can be mitigated by the use of early stop and K-fold. We take the model with the best validation accuracy (around 92%) from the k-fold cross validation as our final model for subimage classification.
6. The resulting Percentages.png suggests that the optimal threshold to be 8 (image classified as pure if 8/12 subimages are classified as pure), which maximizes the average (of 5 types of fertilizers) prediction accuracy at about 96.5% with each individual type above 95%.

To-Do List of Model 2.0:
- [x] Fix bugs in Model 1.0
- [x] DNN reconstruction -- stacking Conv2D & FC layers
- [x] Parameter Tuning -- Kernel & Feature map sizes, Batch Norm, Optimizer
- [x] Exponential decay learning rate schedule
- [x] Customize callback for early stopping -- tracking validation loss decrease & Speak validation accuracy

Lastly, if you have any questions, please contact Wenjie Yu.
