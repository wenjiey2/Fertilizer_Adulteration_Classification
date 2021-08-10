package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.Store;

public class StoreItemAdapter extends RecyclerView.Adapter<StoreItemAdapter.ViewHolder> {

    private OnStoreListener onStoreListener;
    private ArrayList<Store> stores;

    public StoreItemAdapter(OnStoreListener onStoreListener, ArrayList<Store> stores) {
        this.onStoreListener = onStoreListener;
        this.stores = stores;
    }

    @Override
    public StoreItemAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        Context context = parent.getContext();
        LayoutInflater inflater = LayoutInflater.from(context);

        // Inflate the custom layout
        View contactView = inflater.inflate(R.layout.gallery_store_item, parent, false);

        // Return a new holder instance
        ViewHolder viewHolder = new ViewHolder(contactView, onStoreListener);
        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull StoreItemAdapter.ViewHolder holder, int position) {
        Store store = stores.get(position);

        TextView storeTextView = holder.storeTextView;
        storeTextView.setText(store.getName());

        TextView villageTextView = holder.villageTextView;
        villageTextView.setText("village: " + store.getVillage());

        TextView districtTextView = holder.districtTextView;
        districtTextView.setText("district: " + store.getDistrict());
    }

    public interface OnStoreListener {
        void onStoreClick(int position);
    }

    @Override
    public int getItemCount() {
        return stores == null ? 0 : stores.size();
    }

    // Provide a direct reference to each of the views within a data item
    // Used to cache the views within the item layout for fast access
    public class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        // Your holder should contain a member variable
        // for any view that will be set as you render a row
        public TextView storeTextView;
        public TextView villageTextView;
        public TextView districtTextView;
//        public Button selectStoreButton;

        OnStoreListener onStoreListener;

        public ViewHolder(View itemView, OnStoreListener onStoreListener) {
            super(itemView);
            storeTextView = (TextView) itemView.findViewById(R.id.store_name);
            villageTextView = (TextView) itemView.findViewById(R.id.village_name);
            districtTextView = (TextView) itemView.findViewById(R.id.district_name);
            this.onStoreListener = onStoreListener;
            itemView.setOnClickListener(this);
        }

        @Override
        public void onClick(View v) {
            onStoreListener.onStoreClick(getAdapterPosition());
        }
    }
}
