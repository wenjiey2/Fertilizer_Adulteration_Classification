package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Context;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.widget.Button;

import com.google.firebase.auth.FirebaseAuth;

import edu.illinois.fertilizeradulterationdetection.prediction.MainActivity;
import edu.illinois.fertilizeradulterationdetection.R;

public class InitActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final Context mainActivity = this;

        getWindow().getDecorView().getBackground().setDither(true);
        getWindow().setFormat(PixelFormat.RGBA_8888);

        setContentView(R.layout.activity_init);

        Button logIn = findViewById(R.id.init_log_in_button);
        logIn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(mainActivity, LogInActivity.class));
            }
        });

        Button signUp = findViewById(R.id.init_sign_up_button);
        signUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(mainActivity, SignUpActivity.class));
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();

        if (FirebaseAuth.getInstance().getCurrentUser() != null) {
            Intent intent = new Intent(this, MainActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
            startActivity(intent);
        }
    }
}