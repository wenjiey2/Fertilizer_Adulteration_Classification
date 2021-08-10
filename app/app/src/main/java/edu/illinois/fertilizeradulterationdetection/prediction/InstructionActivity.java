package edu.illinois.fertilizeradulterationdetection.prediction;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import edu.illinois.fertilizeradulterationdetection.R;

public class InstructionActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final Context instructionActivity = this;
        setContentView(R.layout.activity_instruction);

        Button next = findViewById(R.id.next);
        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(instructionActivity, MainActivity.class);
                intent.putExtra("next", true);
                startActivity(intent);
            }
        });
    }
}