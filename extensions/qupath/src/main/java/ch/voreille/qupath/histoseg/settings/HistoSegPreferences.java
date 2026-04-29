package ch.voreille.qupath.histoseg.settings;

import java.util.prefs.Preferences;

public class HistoSegPreferences {

    private static final Preferences PREFS = Preferences.userNodeForPackage(HistoSegPreferences.class);

    private static final String KEY_SERVER_URL = "server_url";
    private static final String KEY_MODEL_ID = "model_id";

    public static String getServerUrl() {
        return PREFS.get(KEY_SERVER_URL, "http://localhost:8090");
    }

    public static void setServerUrl(String value) {
        PREFS.put(KEY_SERVER_URL, value);
    }

    public static String getModelId() {
        return PREFS.get(KEY_MODEL_ID, "default");
    }

    public static void setModelId(String value) {
        PREFS.put(KEY_MODEL_ID, value);
    }
}