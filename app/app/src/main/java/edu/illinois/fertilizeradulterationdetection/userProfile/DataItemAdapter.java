package edu.illinois.fertilizeradulterationdetection.userProfile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.whamu2.previewimage.Preview;

import java.io.File;
import java.util.ArrayList;

import edu.illinois.fertilizeradulterationdetection.R;
import edu.illinois.fertilizeradulterationdetection.entity.Image;

public class DataItemAdapter extends RecyclerView.Adapter<DataItemAdapter.ViewHolder> {

    private ArrayList<Image> images;
    private ArrayList<com.whamu2.previewimage.entity.Image> loadImages = new ArrayList<>();
    private Context context;

    public DataItemAdapter(ArrayList<Image> images) {
        this.images = images;

        for (Image image : images) {
            com.whamu2.previewimage.entity.Image i = new com.whamu2.previewimage.entity.Image();
            i.setThumbnailUrl(image.getImagePath());
            loadImages.add(i);
        }
    }

    @Override
    public DataItemAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        context = parent.getContext();
        LayoutInflater inflater = LayoutInflater.from(context);

        // Inflate the custom layout
        View contactView = inflater.inflate(R.layout.gallery_data_item, parent, false);

        // Return a new holder instance
        ViewHolder viewHolder = new ViewHolder(contactView, images);
        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Image image = images.get(position);
        holder.predictionTextView.setText(image.getPrediction());
        holder.dateTextView.setText(image.getDate());

        File imgFile = new File(image.getImagePath());

        if(imgFile.exists()){
            Bitmap myBitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
            holder.imageView.setImageBitmap(myBitmap);
        }
        holder.imageView.setTag(position);
        holder.imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Preview.with(context)
                        .builder()
                        .load(loadImages) // set of image path
                        .displayCount(true) // display total image count
                        .markPosition((int) view.getTag()) // mark image index
                        .showDownload(false) // whether to show download button (when image exists)
                        .showOriginImage(false) // whether to show original image (when image exists)
                        .downloadLocalPath("Preview") // download image path /storage/emulated/0/Pictures/Preview/
                        .show();
            }
        });
    }

    @Override
    public int getItemCount() {
        return images == null ? 0 : images.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        public TextView dateTextView;
        public TextView predictionTextView;
        public ImageView imageView;

        public ViewHolder(View itemView, ArrayList<Image> images) {
            super(itemView);
            dateTextView = (TextView) itemView.findViewById(R.id.instance_date);
            predictionTextView = (TextView) itemView.findViewById(R.id.instance_prediction);
            imageView = (ImageView) itemView.findViewById(R.id.instance_image);
        }
    }


}
