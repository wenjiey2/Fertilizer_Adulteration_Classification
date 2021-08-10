# YOLOv4 "Region of Interest" Detection

To help users provide images of the best quality for inference, we provided the option for users to crop the image. Here, we implemented a "Region of Interest" Detection with YOLOv4, which would set the initial cropping range to further assist our users.

### References
- Original YOLO Paper (State of the Art): https://arxiv.org/pdf/1506.02640.pdf
- YOLOv4 Paper: https://arxiv.org/abs/2004.10934
- YOLOv4 Original Source Code: https://github.com/AlexeyAB/darknet

### Training
We have used darknet for training to generate the .weights files and converted them to tflite files for mobile inference with the options of different quantization levels.

**The current update is trained on YOLOv4-tiny and supports real-time inference on mobile device.** The size of the model is reduced from ~200 MB to ~20 MB for the original .weights file, and from ~61 MB to ~6 MB for the .tflite model. Deployment in our APP would be available in future updates.

#### Labelling Strategy for Training Data
Instead of cropping out 100% of the irrelevance background (sometimes not realizable when noise & fertilizer pixels are not linearly separable), a relatively conservative scheme was used in labelling our training set, in which edges of bounding boxes are pushed inwards until the area of neglected fertilizer is approximately larger than the neglected background. A set of soft criterions are also implemented for specific images depending on how each image looks like. For example, for some images, we have set the tolerance on the number of noisy pixels on the vertical edge to be less than 1/3 of the image height. The resulting region will still contain a certain amount of background, but we can make sure we don't crop off too many pixels of the fertilizer. This also couples with data augmentation in our inference model perfectly when fill mode is constant.

#### Performance on our customized training set
Training was scheduled for 6000 epochs. We have recorded the trained weights every 1000 epochs as checkpoints. Notice we have not yet updated our [Cropping Effectiveness Analysis](https://github.com/ACES-UIUC-Fertilizer-Group/Fertillizer_Adulteration_Detection_app/tree/master/model/model_2.1/YOLOv4#evaluation-of-cropping-effectiveness) for YOLOv4-tiny model since we are in the mix of developing a new model for adulteration detection. For the purpose of the analysis, we prioritized detection (mAP) over accuracy (IoU). For your use case in cropping, you might want to prioritize average IoU for cropping accuracy once you are satisfied with the detection rate. We have provided 4 .weights files as well as the corresponding quantized .tflite files. We strongly recommend trying out all of weight files provided.

| Epochs | mAP@0.50  | average IoU (confidence threshold = 0.25) |
| ------ | ------- | ----------------------------------------- |
| yolov4-obj_1000.weights | 80.12% | 62.79% |
| yolov4-obj_2000.weights | 87.15% | 69.52% |
| yolov4-obj_3000.weights | 89.23% | 73.74% |
| yolov4-obj_4000.weights | 86.84% | 70.35% |
| yolov4-obj_5000.weights | 90.60% | 71.08% |
| yolov4-obj_6000.weights | 88.36% | 71.41% |


### Inference with tflite in Python
- detect.py - detects bounding box(es) for cropping and saves the image name, bounding box coordinates and confidence for cropping in yolo_crop.py
- yolo_crop.py - parse data.json and crop images specified with the pair of coordinates. Notice that if multiple bounding boxes are detected, we choose the one with the highest confidence; if no bounding boxes are detected, the image remains uncropped.

**Our pretrained weights can be found [here](https://drive.google.com/drive/folders/1OWLoztL1UKCEspmvD1kzp6IOnlIfw6nN).**

We have provided tflite models of both floating-point & fixed-point precisions, namely float32, float16 and int8. The int8 model is the most quantized, which saves inference time on the cost of losing accuracy in mAP & average IoU. However, notice that the actual performance coupled with the adulteration detection model may depend on the specific image. In the case of our dataset, fixed-point 8-bit quantization resulted in higher prediction accuracy for FADNet since we adopted a somewhat conservative cropping scheme.

Since our training, testing & analysis were done on PC, we used the baseline YOLOv4 for accuracy. Currently, even with quantization, the floating point models are still larger than 100MB. Therefore, we strongly recommend using the int8 model for inference on mobile devices.

#### Simple Demo
- **To run the demo below, Tensorflow 2.3.0rc0 or above is required.**
- **You also need to have a working Tensorflow version of YOLOv4, links are provided by YOLOv4 author in his github page referenced above. Please place their *core* and *data* folders within our YOLOv4 folder.**
- **Prepare images.txt which contains absolute paths to the set of fertilizer images you want to crop.**
- **Change the output path in detect.py to a valid path on your device accordingly.**
- **Change the image and output path in yolo_crop.py accordingly, i.e. img_path = '*the directory you stored images listed in images.txt*' + key**
- **Move into the directory where the tflite & txt files are located on your device, or change weights and image flag to the absolute path of the tflite & txt file.**
- **Uncomment line 71 to 91 in detect.py if you are using any int8 models.**

**Run the below code in YOLOv4 directory:**

```bash
# Inference with int8 model, use --iou and --score to change the IoU & confidence threshold if you wish
python detect.py --weights ./yolov4-obj_3000-int8.tflite --size 416 --image ./images.txt --framework tflite --tiny
```

**By now, data.json should be created in the same directory.**

```bash
# Crop the images
python yolo_crop.py
```

### Evaluation of Cropping Effectiveness
We ran test.py on all three cropped datasets of different quantization, and compare the accuracy with the uncropped dataset. The detection accuracy of full images can vary depending on the version of FadNet used. However, the accuracy gain of subimage classification is universal, in that cropping with our labelling scheme and trained YOLOv4 weights will always produce roughly 2x pure detection compared to no cropping. For each full image, cropping most likely affects only the 10 boarder images out of a total 12 subimages (extreme cases do exist when YOLOv4 crops very unevenly in either axis), the cropping effectiveness can be even higher than the numbers shown below. Overall, the performance improvement of the model by cropping using "region of interest" detection without complex image preprocessing is very noticeable, considering that the dataset we prepared for detection simulates the very worst input images from a user, with not only background irrelevance, but also blurriness, reflective surfaces, spotlight, and extreme brightness/dimness.

| Quantization Type | Image Type | Accuracy |
| ----------------- | ---------- | -------- |
| No Cropping | Full Images | 25 out of 268 (9.33%) |
| No Cropping | Subimages | 653 out of 3216 (20.30%) |
| float32 | Full Images | 76 out of 268 (28.36%) |
| float32 | Subimages | 1322 out of 3216 (41.11%) |
| float16 | Full Images |76 out of 268 (28.36%)  |
| float16 | Subimages | 1318 out of 3216 (49.18%) |
| int8 | Full Images | 81 out of 268 (30.22%) |
| int8 | Subimages | 1331 out of 3216 (49.66%) |

Lastly, if you have any questions, please contact Wenjie Yu.
