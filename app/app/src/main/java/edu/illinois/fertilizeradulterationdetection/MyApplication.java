package edu.illinois.fertilizeradulterationdetection;

import android.app.Application;

import edu.illinois.fertilizeradulterationdetection.config.Constants;
import edu.illinois.fertilizeradulterationdetection.utils.SharedPreferencesUtils;

public class MyApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        SharedPreferencesUtils.init(getApplicationContext(), Constants.SP_NAME);
    }

}