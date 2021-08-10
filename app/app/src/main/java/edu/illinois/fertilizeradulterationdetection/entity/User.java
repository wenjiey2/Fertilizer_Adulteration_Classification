package edu.illinois.fertilizeradulterationdetection.entity;

import android.content.Context;

import com.google.gson.Gson;

import java.util.ArrayList;
import java.util.Set;

import edu.illinois.fertilizeradulterationdetection.config.Constants;
import edu.illinois.fertilizeradulterationdetection.utils.SharedPreferencesUtils;

public class User {
    private String phoneNumber, email, occupation;
    private static ArrayList<Store> stores = new ArrayList<>();
    public static User user;

    public User(String number, String occ, String em) {
        email = em;
        phoneNumber = number;
        occupation = occ;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public String getEmail() {
        return email;
    }

    public String getOccupation() {
        return occupation;
    }

    public void setPhoneNumber(String number) {
        phoneNumber = number;
    }

    public void setOccupation(String occ) {
        occupation = occ;
    }

    public static ArrayList<Store> getStores(Context context) {
        stores.clear();
        SharedPreferencesUtils spu = SharedPreferencesUtils.init(context);
        Set<String> storeNames = spu.getStringSet(Constants.SP_KEY_STORE_NAMES);
        for (String storeName : storeNames) {
            String storeStr = spu.getString(storeName);
            Store store = new Gson().fromJson(storeStr, Store.class);
            stores.add(store);
        }
        return stores;
    }
}
