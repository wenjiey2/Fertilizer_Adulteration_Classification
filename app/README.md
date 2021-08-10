# Installation

To install the app, please download the [apk file](https://github.com/ACES-UIUC-Fertilizer-Group/Fertillizer_Adulteration_Detection_app/blob/master/app/app/release/app-release.apk) and install it on your android phone.

# App Documentaion

## User Registration/Login

### SignUpActivity

We recommend using phone number for sign up & verification. Additional email registration is enabled, but not user information is not interfaced with the backend yet.

### LoginActivity

Similarly, email login with user password is supported, but not recommended.


### VerificationActivity

###

## Prediction

### MainActivity

This activity allows users to upload a photo by choosing a photo from the gallery, or by taking a photo from the phone's camera.

### InstructionActivity

This activity instructs users on how to take good-quality photos for better prediction accuracy.

### InfoActivity

This activity allows users to enter user-specific metadata, including note, store, village, and district.

### PredictActivity

This activity predicts the input photo as adulterated or pure fertilizer and provides the options of saving the result and generating reports.

# Beta Testing
* We are working with researchers at the International Institute for Tropical Agriculture (IITA) to beta test the app.
* They aren’t receiving any confirmation code so can’t use the app.

## Bugs Fixed
- [x] Sign up and Login with SMS verification crashed the APP for Tanzania users due to incorrect E164 format.
- [x] Email sign up crashes the APP with a reported Firebase Remote Exception.
- [x] In PredictActivity, if the user clicks "report" then "save," the saved prediction is without metadata upon retrieval in "History" in StoreGalleryActivity. If the user clicks “save” then “report,” it will have metadata as intended.
- [x] If I click “report,” photos are saved to two different albums in my folders, FAD and Pictures, need to restructure the storage directories.
- [x] If I click “save,” the prediction is saved to something called “Reports,” which is changed to "History."

## Future Updates in Functionalities:
- [x] Add a first layer to the model to distinguish between fertilizer and non-fertilizer to filter out and not give predictions on non-fertilizer: The YOLOv4 model is developed already to crop out the relevant part of the image that contains the fertilizer, and suggest user to reupload a better-quality image if it could not find any relevant area. Further work needs to be done to deploy the model and interface it with the cropping feature that's yet to be developed.
# Credits

For questions, please contact Wenjie Yu via email at wenjiey2@illinois.edu.
