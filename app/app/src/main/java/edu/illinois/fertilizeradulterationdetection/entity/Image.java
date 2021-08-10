package edu.illinois.fertilizeradulterationdetection.entity;

public class Image {
    private double longitude;
    private double latitude;
    private String date;
    private String prediction;
    private String note;
    private String imagePath;
    private boolean saved2Firebase;

    public Image(){
        longitude = 0;
        latitude = 0;
        date = "";
        prediction = "";
        note = "";
        imagePath = "";
        saved2Firebase = true;
    }

    public Image(String date, String prediction, String note, String imagePath){
        this(0,0,date, prediction, note, imagePath, false);
    }

    public Image(double longitude, double latitude, String date, String prediction, String note, String imagePath, Boolean saved2Firebase) {
        this.longitude = longitude;
        this.latitude = latitude;
        this.date = date;
        this.prediction = prediction;
        this.note = note;
        this.imagePath = imagePath;
        this.saved2Firebase = saved2Firebase;
    }

    public void setNote(String note) {
        this.note = note;
    }

    public String getNote() {
        return note;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public double getLatitude() {
        return latitude;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setSaved2Firebase(boolean saved2Firebase) {
        this.saved2Firebase = saved2Firebase;
    }

    public boolean getSaved2Firebase() {
        return saved2Firebase;
    }
}
