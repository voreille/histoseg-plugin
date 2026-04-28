package ch.voreille.qupath.histoseg.client.dto;

import com.google.gson.JsonObject;

/**
 * Response from a future GET /results/{result_id}.
 *
 * Keep GeoJSON/statistics as JsonObject because they are nested dynamic JSON
 * and are parsed elsewhere into QuPath annotations.
 */
public class ResultResponse {

    public long result_id;
    public String status;

    public JsonObject geojson;
    public JsonObject statistics;

    /**
     * Optional fields if the backend returns paths/metadata.
     */
    public String geojson_path;
    public String statistics_path;

    public ResultResponse() {
    }

    public boolean hasGeoJson() {
        return geojson != null;
    }

    public boolean hasStatistics() {
        return statistics != null;
    }
}
