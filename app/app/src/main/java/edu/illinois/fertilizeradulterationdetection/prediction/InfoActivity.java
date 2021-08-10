package edu.illinois.fertilizeradulterationdetection.prediction;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.utils.ConnectivityUtil;
import android.content.Context;
import android.os.SystemClock;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

class FetchTask extends AsyncTask<Void, String, Object> {
    private ProgressDialog progress;
    private Context context;

    public FetchTask(Context cxt) {
        context = cxt;
        progress = new ProgressDialog(context);
        progress.setMessage("Detection in progress...");
    }

    @Override
    protected void onPreExecute() {
        progress.show();
        super.onPreExecute();
    }

    @Override
    protected Void doInBackground(Void... unused) {
        SystemClock.sleep(5000);
        return null;
    }

    @Override
    protected void onPostExecute(Object result) {
        progress.dismiss();
        super.onPostExecute(result);
    }

    public void setProgressText(String text){
        progress.setMessage(text);
    }
}

public class InfoActivity extends AppCompatActivity {

    private Spinner existingStoreView;
    private LinearLayout newStoreView;
    private Button toPrediction;

    private String id;
    private DatabaseReference databaseRef;
    private String storeName;

    private Intent intent;


    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info);

        initialization();

        findViewById(R.id.radio1).setVisibility(View.GONE);

        if(ConnectivityUtil.isNetworkAvailable(this)) {
            retrieveSavedStores();
        }
    }

    private void initialization() {
        newStoreView = findViewById(R.id.newStore);
        existingStoreView = findViewById(R.id.existingStore);

        id = FirebaseAuth.getInstance().getCurrentUser().getUid();
        databaseRef = FirebaseDatabase.getInstance().getReference();

        // set up background image
        Uri uri = Uri.parse(getIntent().getStringExtra("uri"));
        try {
            int img_rot = readPictureDegree(uri.getPath());
            Bitmap src = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
//            if (src.getWidth() > src.getHeight()){
//                img_rot = 90;
//            }
            src = rotateBitmap(src, img_rot);
            ImageView imageView = findViewById(R.id.info_image);
            imageView.setImageBitmap(src);
        } catch (Exception e) {
            Toast.makeText(this, "Fail to load image",Toast.LENGTH_LONG).show();
        }

        // inflater is used to inflate district, village and store text field into its parent view. Please refer to template_label_edittext.xml for details about "custom" view
        LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        final List<String> labels = new ArrayList<>(Arrays.asList("District", "Village", "Store"));
        final List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            assert inflater != null;
            @SuppressLint("InflateParams") View custom = inflater.inflate(R.layout.template_label_edittext, null);

            TextView tv = custom.findViewById(R.id.label);
            tv.setText(labels.get(i) + ": ");

            EditText et = custom.findViewById(R.id.text);
            int id = View.generateViewId();
            ids.add(id);
            et.setId(id);

            newStoreView.addView(custom);
        }

        //pass info to prediction
        final FetchTask fetchTask = new FetchTask(this);
        intent = new Intent(getApplicationContext(), PredictActivity.class);
        toPrediction = findViewById(R.id.to_prediction);
        toPrediction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // relay extras from previous activity
                intent.putExtra("uri", getIntent().getStringExtra("uri"));
                intent.putExtra("isFromStorage", getIntent().getBooleanExtra("isFromStorage", false));

                // pass new extras obtained in this activity
                EditText noteView = findViewById(R.id.note_text);
                intent.putExtra("note", noteView.getText().toString());

                if (storeName == null) {
                    for (int i = 0; i < 3; i++) {
                        EditText v = findViewById(ids.get(i));
                        if(TextUtils.isEmpty(v.getText())){
                            Toast.makeText(InfoActivity.this, "Store, district and village cannot be null.",Toast.LENGTH_LONG).show();
                            return;
                        }
                        intent.putExtra(labels.get(i), v.getText().toString());
                    }
                } else {
                    intent.putExtra("Store", storeName);
                }
                fetchTask.execute();
                startActivity(intent);
            }
        });
    }

    public void checkButton(View V) {
        RadioGroup radioGroup = findViewById(R.id.radio);;
        int radioId = radioGroup.getCheckedRadioButtonId();
        // save to existing store
        if (radioId == R.id.radio1) {
            existingStoreView.setVisibility(View.VISIBLE);
            newStoreView.setVisibility(View.GONE);
            toPrediction.setVisibility(View.VISIBLE);
        } else { // save to new store
            newStoreView.setVisibility(View.VISIBLE);
            existingStoreView.setVisibility(View.GONE);
            toPrediction.setVisibility(View.VISIBLE);
            intent.putExtra("newStore", true);
        }
    }

    private void retrieveSavedStores() {
        findViewById(R.id.radio1).setVisibility(View.VISIBLE);
        databaseRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                ArrayList<String> images = new ArrayList<>();
                ArrayList<String> stores = new ArrayList<>();
                for(DataSnapshot child : dataSnapshot.child(id).child("stores").getChildren()){
                    stores.add(child.getKey());
                    HashMap<String, Object> imgs = (HashMap<String, Object>) ((HashMap<String, Object>)child.getValue()).get("images");
                    for (String s : imgs.keySet()) {
                        images.add(s);
                    }
                }

                if (images.size() == 0) {
                    findViewById(R.id.radio1).setVisibility(View.GONE);
                }

                ArrayAdapter<String> adapter = new ArrayAdapter<>(getApplicationContext(), R.layout.spinner_item, stores);
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                existingStoreView.setAdapter(adapter);
                existingStoreView.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                        storeName = (String) adapterView.getItemAtPosition(i);
                        intent.putExtra("newStore", false);
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> adapterView) {

                    }
                });
            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {
                throw databaseError.toException();
            }
        });
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
