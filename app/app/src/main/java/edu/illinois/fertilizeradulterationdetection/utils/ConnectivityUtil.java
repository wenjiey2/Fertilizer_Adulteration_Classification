package edu.illinois.fertilizeradulterationdetection.utils;

import android.app.Activity;
import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

public final class ConnectivityUtil {

  public static boolean isNetworkAvailable(Context c) {
    ConnectivityManager connectivityMgr = (ConnectivityManager) c.getSystemService(Context.CONNECTIVITY_SERVICE);
    NetworkInfo networkInfo = connectivityMgr.getActiveNetworkInfo();
    /// if no network is available networkInfo will be null
    if (networkInfo != null && networkInfo.isConnected()) {
      return true;
    }
    return false;
  }
}
