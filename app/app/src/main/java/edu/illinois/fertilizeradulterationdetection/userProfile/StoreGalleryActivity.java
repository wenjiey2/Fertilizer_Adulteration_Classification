package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.RecyclerView;

import com.google.gson.Gson;

import java.util.ArrayList;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.Store;
import edu.illinois.fertilizeradulterationdetection.entity.User;

public class StoreGalleryActivity extends AppCompatActivity implements StoreItemAdapter.OnStoreListener {

    private ArrayList<Store> stores;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_store_gallery);

        // Lookup the recyclerview in activity layout
        RecyclerView recyclerView = (RecyclerView) findViewById(R.id.store_recycler_view);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        recyclerView.setHasFixedSize(true);
        recyclerView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        stores = User.getStores(this);
        StoreItemAdapter adapter = new StoreItemAdapter(this, stores);
        recyclerView.setAdapter(adapter);
    }

    @Override
    public void onStoreClick(int position) {
        String storeJsonStr = new Gson().toJson(stores.get(position));
        Intent intent = new Intent(this, DataGalleryActivity.class);
        intent.putExtra("store_json_str", storeJsonStr);
        startActivity(intent);
    }
}
