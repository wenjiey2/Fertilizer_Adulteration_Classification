package edu.illinois.fertilizeradulterationdetection.prediction;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.media.ExifInterface;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.drew.imaging.ImageMetadataReader;
import com.drew.imaging.ImageProcessingException;
import com.drew.lang.GeoLocation;
import com.drew.metadata.Metadata;
import com.drew.metadata.exif.ExifSubIFDDirectory;
import com.drew.metadata.exif.GpsDirectory;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import com.google.gson.Gson;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.lang.Math;
import java.lang.Object;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.config.Constants;
import edu.illinois.fertilizeradulterationdetection.entity.Image;
import edu.illinois.fertilizeradulterationdetection.entity.Store;
import edu.illinois.fertilizeradulterationdetection.utils.ConnectivityUtil;
import edu.illinois.fertilizeradulterationdetection.utils.SharedPreferencesUtils;
import edu.illinois.fertilizeradulterationdetection.utils.UriUtils;

public class PredictActivity extends AppCompatActivity {

    private static final int WRITE_EXTERNAL_STORAGE_CODE = 1;
    private static int idx = 0;
    private static float purePoss = 0;
    private static int conf_count;
    private static int ad_count;
    private static float pred_conf = 0;

    private Bitmap bitmap;
    private Bitmap bitmap_os;
    private File image;
    private Uri uri;
    private String imagePath;

    private String id;
    private DatabaseReference databaseRef;
    private String prediction;

    private String storeStr, districtStr, villageStr;

    private Interpreter tflite;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private ByteBuffer imgData;
    private static final int X = 320, Y = 320;
    private static final int[] intValues = new int[X * Y];
    private Store str;
    private boolean isSaved = false;
    private boolean isGenerated = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_predict);

        initialization();
        // model preprocessing, first resize the image into 960 * 1280, then slice it into twelve 320 * 320 sub-images.
        bitmap = Bitmap.createScaledBitmap(bitmap, 960, 1280, true);
        final ArrayList<Bitmap> images = splitImage();

        // initialize model & predict
        init_Subimg();

        // Initialize interpreter with NNAPI delegate for acceleration
//        NnApiDelegate nnApiDelegate = null;
//        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            nnApiDelegate = new NnApiDelegate();
//            tfliteOptions.addDelegate(nnApiDelegate);
//            tfliteOptions.setUseNNAPI(true);
//        }

        tfliteOptions.setNumThreads(2 * Runtime.getRuntime().availableProcessors() + 1);
        // Initialize tflite interpreter
        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "pruned_model.tflite"), tfliteOptions);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }

        // Criteria for adulteration: weighting center subimages more than corner ones, (0.067, 0.01, 0.083) for (corner, center, edge) subimages

        prediction = "Adulterated";
        conf_count = 0;
        ad_count = 0;

        for (Bitmap a : images) {
            pred_conf = predict(a);
            if (pred_conf >= 0.75) {
                conf_count += 1;
            } else if (pred_conf <= 0.25) {
                ad_count += 1;
            }

            if (idx == 0 || idx == 3 || idx == 8 || idx == 11) {
                purePoss += pred_conf * 0.071;
            } else if (idx == 5 || idx == 6) {
                purePoss += pred_conf * 0.094;
            } else {
                purePoss += pred_conf * 0.088;
            }

            if (purePoss >= 0.48) {
                prediction = "Pure";
                break;
            }
            idx += 1;
        }

        if (ad_count > 4) {
            prediction = "Adulterated";
        }
        else if (conf_count >= 8) {
            prediction = "Pure";
        }

        // Unload delegate
        tflite.close();
//        if(null != nnApiDelegate) {
//            nnApiDelegate.close();
//        }

        TextView predView = findViewById(R.id.prediction);
        if (((purePoss < 0.6) && (prediction == "Pure")) || ((purePoss > 0.4) && (prediction == "Adulterated"))) {
            predView.setText(prediction + " (low confidence)");
        } else {
            predView.setText(prediction);
        }
        storageInit();
    }

    /**
     * ======================================= Initialization ================================================
     **/

    private void initialization() {
        // retrieve data from previous activity
        Intent intent = getIntent();
        storeStr = intent.getStringExtra("Store");
        districtStr = intent.getStringExtra("District");
        villageStr = intent.getStringExtra("Village");
        uri = Uri.parse(intent.getStringExtra("uri"));

        // background image
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            int img_rot = readPictureDegree(uri.getPath());
            bitmap_os = rotateBitmap(bitmap, img_rot);
            ImageView imageView = findViewById(R.id.image);
            imageView.setImageBitmap(bitmap_os);
        } catch (Exception e) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_LONG).show();
        }

        // get image path of the photo in the phone
        boolean isFromStorage = getIntent().getBooleanExtra("isFromStorage", false);
        if (isFromStorage) {
//            imagePath = getRealPathFromURI(uri);
            File file = UriUtils.uri2File(getApplication(), uri);
            if (file != null) {
                imagePath = UriUtils.uri2File(getApplication(), uri).getPath();
            } else {
                imagePath = "";
                Toast.makeText(this, "Image uploaded from cloud service. We recommend uploading from local album or camera.", Toast.LENGTH_LONG).show();
            }
        } else {
            imagePath = uri.getPath();
        }
    }

    private void storageInit() {
        // button initialization
        final Button save = findViewById(R.id.save);
        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // check wifi connectivity to determine storage location
                boolean isConnected = ConnectivityUtil.isNetworkAvailable(view.getContext());
                if (ContextCompat.checkSelfPermission(PredictActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(PredictActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, WRITE_EXTERNAL_STORAGE_CODE);
                }
                if (!isSaved) {
                    saveImageAttr(isConnected);
                    isSaved = true;
                } else {
                    Toast.makeText(getApplicationContext(), "Image already saved.", Toast.LENGTH_LONG).show();
                    save.setEnabled(false);
                }
            }
        });

        final Button report = findViewById(R.id.report);
        report.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (ContextCompat.checkSelfPermission(PredictActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(PredictActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, WRITE_EXTERNAL_STORAGE_CODE);
                } else {
                    if (!isGenerated) {
                        isGenerated = true;
                        generateReport();
                    } else {
                        Toast.makeText(getApplicationContext(), "Report already saved.", Toast.LENGTH_LONG).show();
                        report.setEnabled(false);
                    }
                }
            }
        });

        Button back = findViewById(R.id.back);
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(getApplicationContext(), MainActivity.class));
            }
        });

        // firebase initialization
        id = FirebaseAuth.getInstance().getCurrentUser().getUid();
        databaseRef = FirebaseDatabase.getInstance().getReference();
    }

    //code from stackOverflow: https://stackoverflow.com/a/23920731/12234267
//    private String getRealPathFromURI(Uri uri) {
//
//        String filePath = "";
//        String wholeID = DocumentsContract.getDocumentId(uri);
//
//        String id = wholeID.split(":")[1];
//        String[] column = {MediaStore.Images.Media.DATA};
//
//        String sel = MediaStore.Images.Media._ID + "=?";
//        Cursor cursor = this.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
//                column, sel, new String[]{id}, null);
//
//        assert cursor != null;
//        int columnIndex = cursor.getColumnIndex(column[0]);
//
//        if (cursor.moveToFirst()) {
//            filePath = cursor.getString(columnIndex);
//        }
//        cursor.close();
//        return filePath;
//    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == WRITE_EXTERNAL_STORAGE_CODE) {
            //generateReport();
        }
    }

    /**
     * ======================================= Model ================================================
     **/

    // Split each image into 12 sub-images to increase robustness
    private ArrayList<Bitmap> splitImage() {
        final int rows = 4;
        final int columns = 3;
        final int chunks = rows * columns;
        final int height = bitmap.getHeight() / rows;
        final int width = bitmap.getWidth() / columns;

        ArrayList<Bitmap> images = new ArrayList<>(chunks);

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, bitmap.getWidth(), bitmap.getHeight(), true);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                int xCoord = j * width, yCoord = i * height;
                Bitmap image = Bitmap.createBitmap(scaledBitmap, xCoord, yCoord, width, height);
                images.add(image);
            }
        }

        return images;
    }

    // Please refer to the demo from TensorFlow official github site:
    // run "git clone https://www.github.com/tensorflow/tensorflow" and find its demo at /tensorflow/lite/java/demo
    // Useful link: https://medium.com/tensorflow/using-tensorflow-lite-on-android-9bbc9cb7d69d
    private void init_Subimg() {
        final int DIM_BATCH_SIZE = 1, DIM_PIXEL_SIZE = 3, NumBytesPerChannel = 4;

        imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE
                        * X
                        * Y
                        * DIM_PIXEL_SIZE
                        * NumBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());
    }

    // convert bitmap of each sub-image to ByteBuffer to feed into the TensorFlowLite.
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < X; ++i) {
            for (int j = 0; j < Y; ++j) {
                final int val = intValues[pixel++];

                imgData.putFloat(((val >> 16) & 0xFF) / 255.f);
                imgData.putFloat(((val >> 8) & 0xFF) / 255.f);
                imgData.putFloat((val & 0xFF) / 255.f);
            }
        }
    }

    // return unweighted possibility of the subimage being pure in fixed point
    private float predict(Bitmap image) {
        convertBitmapToByteBuffer(image);

        float[][] result = new float[1][1];

        if (tflite != null) {
            tflite.run(imgData, result);
        }
        return result[0][0];
    }

    /**
     * ====================================== Save to Database =================================================
     **/
    // save metadata to local SharedPreferences, maybe database after
    private void savedToJson(Image img, Store str, boolean success) {
        img.setSaved2Firebase(success);
        str.addImage(img);
        SharedPreferencesUtils.init(this).put(str.getName(), new Gson().toJson(str, Store.class));
    }

    private void saveImageAttr(boolean isConnected) {

        final ProgressDialog progressDialog1 = new ProgressDialog(this);
        progressDialog1.setMessage("Uploading Image...");
        //progressDialog.setCancelable(false);
        //progressDialog.setCanceledOnTouchOutside(false);
        progressDialog1.show();

        // Create info object
        final Image img = new Image();
        img.setPrediction(prediction);
        img.setNote(getIntent().getStringExtra("note"));

        // Retrieve image metadata
        img.setLongitude(-1);
        img.setLatitude(-1);
        if (imagePath != "") {
            image = new File(imagePath);
            Metadata metadata = null;
            try {
                metadata = ImageMetadataReader.readMetadata(image);
            } catch (ImageProcessingException | IOException e) {
                e.printStackTrace();
            }
            assert metadata != null;

            // date
            ExifSubIFDDirectory exifSubIFDDirectory = metadata.getFirstDirectoryOfType(ExifSubIFDDirectory.class);
            if (exifSubIFDDirectory != null) {
                String date = exifSubIFDDirectory.getDate(ExifSubIFDDirectory.TAG_DATETIME_ORIGINAL).toString();
                img.setDate(date);
            }

            // longitude & latitude
            GpsDirectory gpsDirectory = metadata.getFirstDirectoryOfType(GpsDirectory.class);
            if (gpsDirectory != null) {
                GeoLocation location = gpsDirectory.getGeoLocation();
                if (location != null) {
                    img.setLatitude(location.getLatitude());
                    img.setLongitude(location.getLongitude());
                }
            }

        } else {
            img.setDate(new Date().toLocaleString());
        }

        // ========== Save Image internally ========== //
        boolean newStore = getIntent().getBooleanExtra("newStore", true);

        //new Store
        if (newStore) {
            str = new Store(storeStr, districtStr, villageStr);
        } else {//old store
            String json = SharedPreferencesUtils.init(this).getString(storeStr, "");
            if (TextUtils.isEmpty(json)) {
                str = new Store(storeStr, districtStr, villageStr);
            } else {
                str = new Gson().fromJson(json, Store.class);
            }
        }
        Set<String> stringSet = SharedPreferencesUtils.init(this).getStringSet(Constants.SP_KEY_STORE_NAMES);
        stringSet.add(storeStr);
        SharedPreferencesUtils.init(this).putStringSet(Constants.SP_KEY_STORE_NAMES, stringSet);

        // ========== Save to Firebase ========== //

        String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath();
        File FADFolder = new File(rootPath + "/FAD/Raw Image/");
        if (!FADFolder.exists()) {
            FADFolder.mkdirs();
        }
        try {
            String fileName = UUID.randomUUID().toString();
            imagePath = rootPath + "/FAD/Raw Image/" + fileName + ".jpg";
            img.setImagePath(imagePath);
            File file = new File(imagePath);
            FileOutputStream out = new FileOutputStream(file);
            bitmap_os.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            isSaved = false;
            e.printStackTrace();
        }

        if (!isConnected) {
            progressDialog1.dismiss();
            Toast.makeText(getApplicationContext(), "No WIFI Connection! Image saved to local album.", Toast.LENGTH_LONG).show();
            //save to local on no network
            savedToJson(img, str, false);
        } else {
            DatabaseReference storeRef = databaseRef.child(id).child("stores").child(storeStr);
            if (getIntent().hasExtra("District")) {
                storeRef.child("district").setValue(districtStr);
                storeRef.child("village").setValue(villageStr);
            }
            DatabaseReference imageRef = storeRef.child("images");
            String imageId = imageRef.push().getKey();
            imageRef.child(Objects.requireNonNull(imageId)).setValue(img);

            // save image to storage
            StorageReference storageRef = FirebaseStorage.getInstance().getReference().child(imageId);
            storageRef.putFile(uri).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                    progressDialog1.dismiss();
                    Toast.makeText(getApplicationContext(), "Upload Success!", Toast.LENGTH_SHORT).show();
                    //save to local on upload success
                    savedToJson(img, str, true);
                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    progressDialog1.dismiss();
                    Toast.makeText(getApplicationContext(), "Upload Failed!", Toast.LENGTH_SHORT).show();
                    //save to local on failure
                    savedToJson(img, str, false);
                }
            });
        }
    }

    /**
     * ====================================== Generate Report =================================================
     **/

    private void generateReport() {

        if (imagePath == "") {
            Toast.makeText(getApplicationContext(), "Image from cloud needs to be saved before generating a report.", Toast.LENGTH_LONG).show();
            isGenerated = false;
            return;
        }

        int img_rot = readPictureDegree(imagePath);
        Bitmap src = BitmapFactory.decodeFile(imagePath);
        src = rotateBitmap(src, img_rot);

        Bitmap dest = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);

        Canvas cs = new Canvas(dest);
        Paint tPaint = new Paint();

        //draw background image
        cs.drawBitmap(src, 0f, 0f, null);

        //draw black background
        tPaint.setTextSize((float) Math.floor(src.getHeight() / 25));
        tPaint.setStyle(Paint.Style.FILL);
        float height = tPaint.measureText("yY");
        tPaint.setColor(Color.TRANSPARENT);
        cs.drawRect(0, src.getHeight() - height * 5, src.getWidth(), src.getHeight(), tPaint);

        //write prediction
        tPaint.setColor(Color.WHITE);
        float pred_y = src.getHeight() - 4 * height;
        cs.drawText("prediction: " + prediction, (src.getWidth() - tPaint.measureText("prediction: " + prediction)) / 2, pred_y, tPaint);

        //write store
        tPaint.setTextSize((float) Math.floor(src.getHeight() / 30));
        tPaint.setStyle(Paint.Style.FILL);
        height = tPaint.measureText("yY");
        tPaint.setColor(Color.WHITE);
        cs.drawText("district: " + districtStr + "   store: " + storeStr, (src.getWidth() - tPaint.measureText("district: " + districtStr + "   store: " + storeStr)) / 2, pred_y + 2 * height, tPaint);

        //write date
        tPaint.setStyle(Paint.Style.FILL);
        tPaint.setColor(Color.WHITE);
        String date = new SimpleDateFormat("MM/dd/yyyy", Locale.getDefault()).format(new Date());
        cs.drawText("date generated: " + date, (src.getWidth() - tPaint.measureText("date generated: " + date)) / 2, pred_y + 4 * height, tPaint);

        String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath();
        File ReportsFolder = new File(rootPath + "/FAD/Reports/");
        if (!ReportsFolder.exists()) {
            ReportsFolder.mkdirs();
        }
        try {
            String fileName = UUID.randomUUID().toString();
            imagePath = rootPath + "/FAD/Reports/" + fileName + ".jpg";
            File file = new File(imagePath);
            FileOutputStream out = new FileOutputStream(file);
            dest.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
            Toast.makeText(this, "Report Saved", Toast.LENGTH_LONG).show();

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to generate report:" + e.getMessage(), Toast.LENGTH_LONG).show();
            isGenerated = false;
        }

    }

    // check rotation
    public static int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int rotate) {
        if (bitmap == null)
            return null;
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        Matrix mtx = new Matrix();
        mtx.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, w, h, mtx, true);
    }
}
