package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;

import android.widget.Spinner;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.CountryData;

public class LogInActivity extends AppCompatActivity {
    private FirebaseAuth auth;
    private String phoneNumber, email, pw;
    private Spinner countrySpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_log_in);

        // Initialize Firebase Auth
        auth = FirebaseAuth.getInstance();

        //country code spinner
        countrySpinner = (Spinner) findViewById(R.id.spinnerCountries);
        ArrayAdapter<String> countrySpinnerAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, CountryData.countryAreaCodes);
        countrySpinner.setAdapter(countrySpinnerAdapter);
        countrySpinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        countrySpinner.setAdapter(countrySpinnerAdapter);
        countrySpinnerAdapter.notifyDataSetChanged();
        countrySpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {

            @Override
            public void onItemSelected(AdapterView<?> adapter, View v,
                                       int position, long id) {
                // On selecting a spinner item
                String item = adapter.getItemAtPosition(position).toString();

                // Showing selected spinner item
                Toast.makeText(getApplicationContext(), "Selected Country : " + CountryData.countryNames[countrySpinner.getSelectedItemPosition()], Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
                // TODO Auto-generated method stub

            }
        });

        //submit button
        Button submitPhone = findViewById(R.id.profile_submit_button_text);
        Button submit_email = findViewById(R.id.profile_submit_button_text2);
        submitPhone.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                signIn(false);
            }
        });
        submit_email.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                signIn(true);
            }
        });
    }

    public void signIn(boolean login_by_email) {
        if (!login_by_email) {
            phoneNumber = ((EditText) findViewById(R.id.user_phone_number_input)).getText().toString().trim();

            if (phoneNumber.isEmpty() || phoneNumber.length() < 8 || !phoneNumber.matches("[0-9]+")) {
                Toast.makeText(this, "Please enter a valid phone number", Toast.LENGTH_LONG).show();
                return;
            }
            auth.useAppLanguage();
            String code = "+" + CountryData.countryAreaCodes[countrySpinner.getSelectedItemPosition()];
            phoneNumber = code + phoneNumber;
        } else {
            email = ((EditText) findViewById(R.id.user_email_input)).getText().toString().trim();
            pw = ((EditText) findViewById(R.id.user_pw_input)).getText().toString().trim();
            if (pw.isEmpty()) {
                Toast.makeText(this, "Password cannot be empty.", Toast.LENGTH_SHORT).show();
                return;
            } else if (pw.length() < 6 || pw.length() > 12){
                Toast.makeText(this, "Please enter a password of length between 6 and 12.", Toast.LENGTH_SHORT).show();
                return;
            } else if (pw.contains(" ")){
                Toast.makeText(this, "Password cannot contain any spaces.", Toast.LENGTH_SHORT).show();
                return;
            }
        }
        Intent intent = new Intent(this, VerificationActivity.class);
        intent.putExtra("phoneNumber", phoneNumber);
        intent.putExtra("email", email);
        intent.putExtra("pw", pw);
        intent.putExtra("registrationMethod", login_by_email);
        startActivity(intent);
//        DatabaseReference ref = FirebaseDatabase.getInstance().getReference();
//
//        final String username = ((EditText)findViewById(R.id.user_username_input)).getText().toString().trim();
//        final String password = ((EditText)findViewById(R.id.user_password_input)).getText().toString().trim();
//
//        ref.addValueEventListener(new ValueEventListener() {
//            @Override
//            public void onDataChange(DataSnapshot dataSnapshot) {
//                for(DataSnapshot child : dataSnapshot.getChildren() ){
//                    // Do magic here
//                    User user = child.getValue(User.class);
//                    Log.i("tag","child " + user.getUsername());
//
//                    if(user != null && user.getUsername().equals(username) && user.getPassword().equals(password)){
//                        Log.i("tag","username" + username);
//                        Log.i("tag","password" + password);
//
//                        Intent intent = new Intent(getApplicationContext(), MainActivity.class);
//                        intent.putExtra("username", user.getUsername());
//                        startActivity(intent);
//                    }
//                }
//            }
//
//            @Override
//            public void onCancelled(@NonNull DatabaseError databaseError) {
//
//            }
//
//        });
//        FirebaseAuth mAuth = FirebaseAuth.getInstance();
//        mAuth.signInWithCredential(credential)
//                .addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
//                    @Override
//                    public void onComplete(@NonNull Task<AuthResult> task) {
//                        if (task.isSuccessful()) {
//
//                            FirebaseUser user = task.getResult().getUser();
//                            Log.d("Sign in with phone auth", "Success " + user);
//                            showMainActivity();
//                        } else {
//                            notifyUserAndRetry("Your Phone Number Verification is failed.Retry again!");
//                        }
//                    }
//                });
    }
}
