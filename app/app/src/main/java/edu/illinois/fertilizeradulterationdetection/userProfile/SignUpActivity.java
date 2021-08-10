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

public class SignUpActivity extends AppCompatActivity {
    private FirebaseAuth auth;
    private String phoneNumber, email, pw, occupation;
    private Spinner countrySpinner;

//    @Override
//    protected void onStart() {
//        super.onStart();
//        if (auth.getCurrentUser() != null) {
//            //handle the already login user
//        }
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);

        // Initialize Firebase Auth
        auth = FirebaseAuth.getInstance();

        //set up spinner
        Spinner occupations_spinner = (Spinner) findViewById(R.id.user_occupations_spinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.user_occupations_array, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        occupations_spinner.setAdapter(adapter);
        occupations_spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                occupation = (String) parent.getItemAtPosition(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

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
        Button submit_phone_number = findViewById(R.id.profile_submit_button_text);
        Button submit_email = findViewById(R.id.profile_submit_button_email);
        submit_phone_number.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                registerUser(false);
            }
        });
        submit_email.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                registerUser(true);
            }
        });
    }

    private void registerUser(boolean register_by_email) {
        // When register_by_email = 0, registration is done by text. When register_by_email = 1, registration is done by email.
        Intent intent = new Intent(this, VerificationActivity.class);
        intent.putExtra("occupation", occupation);
        if (!register_by_email) {
            phoneNumber = ((EditText) findViewById(R.id.user_phone_number_input)).getText().toString().trim();
            if (phoneNumber.isEmpty() || phoneNumber.length() != 10 || !phoneNumber.matches("[0-9]+")) {
                Toast.makeText(this, "Please enter a valid phone number", Toast.LENGTH_SHORT).show();
                return;
            }
            auth.useAppLanguage();
            String code = "+" + countrySpinner.getSelectedItemPosition();
            phoneNumber = code + phoneNumber;
            intent.putExtra("phoneNumber", phoneNumber);
        } else {
            email = ((EditText) findViewById(R.id.user_email_input)).getText().toString().trim();
            pw = ((EditText) findViewById(R.id.user_pw_input_su)).getText().toString().trim();
            if (email.isEmpty() || !email.contains("@") || email.length() < 5) {
                Toast.makeText(this, "Please enter a valid email address", Toast.LENGTH_SHORT).show();
                return;
            } else if (pw.length() < 6 || pw.length() > 12) {
                Toast.makeText(this, "Please enter a password of length between 6 and 12.", Toast.LENGTH_SHORT).show();
            } else if (pw.contains(" ")) {
                Toast.makeText(this, "Password cannot contain any spaces.", Toast.LENGTH_SHORT).show();
            }
            intent.putExtra("email", email);
            intent.putExtra("pw", pw);
        }
        intent.putExtra("registrationMethod", register_by_email);
        intent.putExtra("signup", true);
        startActivity(intent);

//        Log.i("tag","phone" + phoneNumber);
//        Log.i("tag","password" + password);
//        Log.i("tag","occupation" + occupation);

//        store username and password in INTERNAL STORAGE
//        String FILENAME = "user";
//        String phoneNumberToWrite = "phone="+phoneNumber;
//        String passwordToWrite = "password="+password;
//        String occupationToWrite= "occupation="+occupation;
//
//        FileOutputStream fstream = null;
//        try {
//            fstream = openFileOutput(FILENAME, Context.MODE_PRIVATE);
//            fstream.write(phoneNumberToWrite.getBytes());
//            fstream.write(passwordToWrite.getBytes());
//            fstream.write(occupationToWrite.getBytes());
//            fstream.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
//
//    //Change UI according to user data.
//    public void  updateUI(FirebaseUser user){
//        if(user != null){
//            Toast.makeText(this,"Sign In successfully",Toast.LENGTH_LONG).show();
//
//            Intent intent = new Intent(this, MainActivity.class);
//            intent.putExtra("username", user.getPhoneNumber());
//            startActivity(intent);
//        }else {
//            Toast.makeText(this,"Please try again",Toast.LENGTH_LONG).show();
//        }
//    }
}
