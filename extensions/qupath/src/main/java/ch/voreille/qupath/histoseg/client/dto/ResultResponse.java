package ch.voreille.qupath.histoseg.client.dto;

import com.google.gson.JsonObject;

public class ResultResponse {

    public String coords_space;
    public JsonObject tissue;
    public JsonObject outputs;
    public JsonObject statistics;

    public ResultResponse() {
    }

    public boolean hasTissue() {
        return tissue != null;
    }

    public boolean hasOutputs() {
        return outputs != null;
    }

    public boolean hasStatistics() {
        return statistics != null;
    }

    public JsonObject toJsonObject() {
        JsonObject obj = new JsonObject();

        if (coords_space != null) {
            obj.addProperty("coords_space", coords_space);
        }

        if (tissue != null) {
            obj.add("tissue", tissue);
        }

        if (outputs != null) {
            obj.add("outputs", outputs);
        }

        if (statistics != null) {
            obj.add("statistics", statistics);
        }

        return obj;
    }
}