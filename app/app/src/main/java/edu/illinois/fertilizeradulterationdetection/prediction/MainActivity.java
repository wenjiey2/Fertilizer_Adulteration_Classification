package edu.illinois.fertilizeradulterationdetection.prediction;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.os.Environment;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.Store;
import edu.illinois.fertilizeradulterationdetection.userProfile.StoreGalleryActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;


public class MainActivity extends AppCompatActivity {
    private final int PICK_IMAGE = 1;
    private final int TAKE_PICTURE = 2;
    private final int READ_EXTERNAL_STORAGE_CODE = 3;
    private final int ACCESS_LOCATION_CODE = 4;
    private Uri uri;

    private Bitmap bitmap;
    private Bitmap bitmap_os;

    private String imagePath;
    private Button toMain;
    private Button toInstruction;
    private Intent intent;

    private Interpreter tflite;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private ByteBuffer imgData;
    private static final int X = 320, Y = 320;
    private static final int[] intValues = new int[X * Y];
    private Store str;
    private final boolean isSaved = false;
    private boolean isGenerated = false;
    private static float conf = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().getDecorView().getBackground().setDither(true);
        getWindow().setFormat(PixelFormat.RGBA_8888);

        setContentView(R.layout.activity_main);

//        User.user = new User("4087570633", "farmer", "");

        // set up reports view
        Button reports = findViewById(R.id.reports);
        reports.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(getApplicationContext(), StoreGalleryActivity.class));
            }
        });

        // set up photo view
        Button photo = findViewById(R.id.photo);
        photo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, READ_EXTERNAL_STORAGE_CODE);
                } else {
                    pickImage();
                }
            }
        });

        // set up camera view
        Button camera = findViewById(R.id.camera);
        StrictMode.VmPolicy.Builder builder = new StrictMode.VmPolicy.Builder();
        StrictMode.setVmPolicy(builder.build());

        camera.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                String date = new SimpleDateFormat("yyyyMMdd", Locale.getDefault()).format(new Date());
                String name = "fertilizer-" + date + "-" + UUID.randomUUID().toString() + ".jpg";
                File photo = new File(getExternalFilesDir(Environment.getDataDirectory().getAbsolutePath()).getAbsolutePath(), name);
                uri = Uri.fromFile(photo);

                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
                if (intent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(intent, TAKE_PICTURE);
                }
            }
        });

        // set up instruction view
        ImageButton info = findViewById(R.id.info);
        info.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(getApplicationContext(), InstructionActivity.class));
            }
        });
    }

    private void pickImage() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == READ_EXTERNAL_STORAGE_CODE) {
            pickImage();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        final FetchTask fetchTask = new FetchTask(this);
        fetchTask.setProgressText("Uploading...");
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == PICK_IMAGE) {
                uri = data.getData();
            }
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // model preprocessing, first resize the image into 320 * 320, then slice it into twelve 320 * 320 sub-images.
            bitmap = Bitmap.createScaledBitmap(bitmap, 320, 320, true);

            // Initialize interpreter with NNAPI delegate for acceleration
//            NnApiDelegate nnApiDelegate = null;
//            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//                nnApiDelegate = new NnApiDelegate();
//                tfliteOptions.addDelegate(nnApiDelegate);
//                tfliteOptions.setUseNNAPI(true);
//            }

            tfliteOptions.setNumThreads(2 * Runtime.getRuntime().availableProcessors() + 1);
            // Initialize tflite interpreter
            try {
                tflite = new Interpreter(FileUtil.loadMappedFile(this, "pruned_fnf_model.tflite"), tfliteOptions);
            } catch (IOException e) {
                Log.e("tfliteSupport", "Error reading model", e);
            }

            conf = 1 - predict(bitmap);

            // Unload delegate
            tflite.close();
//            if(null != nnApiDelegate) {
//                nnApiDelegate.close();
//            }

            TextView predView = findViewById(R.id.prediction);
            if (conf <= 0.5) {
                intent = new Intent(this, FnfActivity.class);
            }
            else{
                intent = new Intent(this, InfoActivity.class);
            }

            intent.putExtra("isFromStorage", requestCode == PICK_IMAGE);
            intent.putExtra("uri", uri.toString());
            intent.putExtra("username", getIntent().getStringExtra("username"));

            fetchTask.execute();

            startActivity(intent);
        }
    }

    private float predict(Bitmap image) {
        convertBitmapToByteBuffer(image);

        float[][] result = new float[1][1];

        if (tflite != null) {
            tflite.run(imgData, result);
        }
        return result[0][0];
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        final int DIM_BATCH_SIZE = 1, DIM_PIXEL_SIZE = 3, NumBytesPerChannel = 4;

        imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE
                        * X
                        * Y
                        * DIM_PIXEL_SIZE
                        * NumBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());

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
}
