package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.Store;

public class DataGalleryActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String storeJsonStr = getIntent().getStringExtra("store_json_str");
        Store store = new Gson().fromJson(storeJsonStr, Store.class);

        setContentView(R.layout.activity_data_gallery);

        RecyclerView recyclerView = (RecyclerView) findViewById(R.id.data_recycler_view);
        DataItemAdapter adapter = new DataItemAdapter(store.getImages());
        recyclerView.setAdapter(adapter);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        recyclerView.setHasFixedSize(true);
        recyclerView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        TextView selected_store = findViewById(R.id.selected_store_name);
        selected_store.setText(store.getName());
        TextView selected_village = findViewById(R.id.selected_village_name);
        selected_village.setText(store.getVillage());
        TextView selected_district = findViewById(R.id.selected_district_name);
        selected_district.setText(store.getDistrict());

    }

}
