package edu.illinois.fertilizeradulterationdetection.prediction;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

import edu.illinois.fertilizeradulterationdetection.R;

public class FnfActivity extends AppCompatActivity {

    private Bitmap bitmap;
    private Bitmap bitmap_os;
    private Uri uri;
    private String warning;

    private Button toMain;
    private Button toInstruction;
    private Intent intent;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_fnf);
        // retrieve data from previous activity
        intent = getIntent();
        uri = Uri.parse(intent.getStringExtra("uri"));

        // background image
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            int img_rot = PredictActivity.readPictureDegree(uri.getPath());
            bitmap_os = PredictActivity.rotateBitmap(bitmap, img_rot);
            ImageView imageView = findViewById(R.id.image);
            imageView.setImageBitmap(bitmap_os);
//            TextView txtView = findViewById(R.id.txt);
//            RelativeLayout.LayoutParams params = new RelativeLayout.LayoutParams(
//                    RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
//            params.topMargin  = 400;
//            txtView.setLayoutParams(params);
        } catch (Exception e) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_LONG).show();
        }

        // button initialization
        toMain = findViewById(R.id.toMain);
        toMain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // jump to sample image
                intent = new Intent(getApplicationContext(), MainActivity.class);
                startActivity(intent);
            }
        });

        toInstruction = findViewById(R.id.toInstruction);
        toInstruction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // jump back to main activity
                intent = new Intent(getApplicationContext(), InstructionActivity.class);
                startActivity(intent);
            }
        });
    }
}
