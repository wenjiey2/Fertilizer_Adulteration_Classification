package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Intent;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseException;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.PhoneAuthCredential;
import com.google.firebase.auth.PhoneAuthOptions;
import com.google.firebase.auth.PhoneAuthProvider;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.concurrent.TimeUnit;

import edu.illinois.fertilizeradulterationdetection.prediction.MainActivity;
import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.User;

public class VerificationActivity extends AppCompatActivity {

    private String verificationId;
    private FirebaseAuth mAuth;
    private ProgressBar progressBar;
    private EditText editText;
    public static String phoneNumber;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        boolean registration_method = getIntent().getBooleanExtra("registrationMethod", false);
        if (!registration_method) {
            setContentView(R.layout.activity_verification);
        } else {
            setContentView(R.layout.activity_verification2);
        }

        mAuth = FirebaseAuth.getInstance();

        progressBar = findViewById(R.id.progressbar);
        editText = findViewById(R.id.editTextCode);

        phoneNumber = getIntent().getStringExtra("phoneNumber");
        String email = getIntent().getStringExtra("email");
        String pw = getIntent().getStringExtra("pw");
        boolean signup = getIntent().getBooleanExtra("signup", false);
        if (!registration_method) {
            PhoneAuthOptions.Builder builder = PhoneAuthOptions.newBuilder();
            builder.setPhoneNumber(phoneNumber);
            builder.setTimeout(60L, TimeUnit.SECONDS);
            builder.setActivity(this);
            builder.setCallbacks(mCallBack);
            PhoneAuthProvider.verifyPhoneNumber(builder.build());
            findViewById(R.id.buttonSignIn).setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {

                    String code = editText.getText().toString().trim();
                    if (code.isEmpty() || code.length() < 6) {
                        editText.setError("Enter code...");
                        editText.requestFocus();
                        return;
                    }
                    verifyCode(code);
                }
            });
        } else {
            emailVerification(email, pw, signup);
        }
    }

    private void verifyCode(String code) {
        PhoneAuthCredential credential = PhoneAuthProvider.getCredential(verificationId, code);
        signInWithCredential(credential);
    }

    private void signInWithCredential(final PhoneAuthCredential credential) {
        mAuth.signInWithCredential(credential)
                .addOnCompleteListener(new OnCompleteListener<AuthResult>() {
                    @Override
                    public void onComplete(@NonNull Task<AuthResult> task) {
                        if (task.isSuccessful()) {
                            if (getIntent().getStringExtra("occupation") != null) {
                                // edit USER
                                User user = new User(getIntent().getStringExtra("phoneNumber"), getIntent().getStringExtra("occupation"), getIntent().getStringExtra("email"));

                                //creates instance in database
                                DatabaseReference ref = FirebaseDatabase.getInstance().getReference();
                                String uid = FirebaseAuth.getInstance().getCurrentUser().getUid();
                                ref.child(uid).setValue(user);
                            }

                            Intent intent = new Intent(VerificationActivity.this, MainActivity.class);
                            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
                            Toast.makeText(VerificationActivity.this, "Verification Complete", Toast.LENGTH_SHORT).show();

                            startActivity(intent);

                        } else {
                            Toast.makeText(VerificationActivity.this, "ERROR", Toast.LENGTH_SHORT).show();
                            Toast.makeText(VerificationActivity.this, task.getException().getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    }
                });
    }

    private void emailVerification(final String email, final String pw, final boolean sign_up) {
        progressBar.setVisibility(View.VISIBLE);
        if (sign_up) {
            mAuth.createUserWithEmailAndPassword(email, pw).addOnCompleteListener(new OnCompleteListener<AuthResult>() {
                @Override
                public void onComplete(@NonNull Task<AuthResult> task) {
                    Intent intent = new Intent(VerificationActivity.this, MainActivity.class);
                    if (!task.isSuccessful()) {
                        intent = new Intent(VerificationActivity.this, SignUpActivity.class);
                        Exception exc = task.getException();
                        try {
                            Toast.makeText(VerificationActivity.this, exc.getMessage(),
                                    Toast.LENGTH_SHORT).show();
                            throw exc;
                        } catch (Exception e) {
                            Toast.makeText(VerificationActivity.this, exc.getMessage(),
                                    Toast.LENGTH_SHORT).show();
                        }
                    }
                    startActivity(intent);

                }
            });
        } else {
            mAuth.signInWithEmailAndPassword(email, pw).addOnCompleteListener(new OnCompleteListener<AuthResult>() {
                @Override
                public void onComplete(@NonNull Task<AuthResult> task) {
                    Intent intent = new Intent(VerificationActivity.this, MainActivity.class);
                    if (!task.isSuccessful()) {
                        intent = new Intent(VerificationActivity.this, InitActivity.class);
                        Exception exc = task.getException();
                        if (exc.getMessage().contains("password is invalid")){
                            intent = new Intent(VerificationActivity.this, LogInActivity.class);
                        }
                        try {
                            throw exc;
                        } catch (Exception e) {
                            Toast.makeText(VerificationActivity.this, exc.getMessage(),
                                    Toast.LENGTH_LONG).show();
                        }
                    }
                    startActivity(intent);
                }
            });
        }
    }

    private PhoneAuthProvider.OnVerificationStateChangedCallbacks mCallBack = new PhoneAuthProvider.OnVerificationStateChangedCallbacks() {

        @Override
        public void onCodeSent(String s, PhoneAuthProvider.ForceResendingToken forceResendingToken) {
            super.onCodeSent(s, forceResendingToken);
            Toast.makeText(getBaseContext(), "code sent", Toast.LENGTH_LONG).show();
            verificationId = s;
        }

        @Override
        public void onVerificationCompleted(PhoneAuthCredential phoneAuthCredential) {
            String code = phoneAuthCredential.getSmsCode();
            if (code != null) {
                editText.setText(code);
                verifyCode(code);
            }
        }

        @Override
        public void onVerificationFailed(FirebaseException e) {
            Toast.makeText(VerificationActivity.this, e.getMessage(), Toast.LENGTH_LONG).show();
        }
    };
}
