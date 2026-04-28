package ch.voreille.qupath.histoseg.client.dto;

import java.util.HashMap;
import java.util.Map;

/**
 * Planned request body for POST /results/lookup.
 *
 * This should match the same parameters used to submit a job, so the backend
 * can compute the same task hash/cache key.
 */
public class ResultLookupRequest {

    public String slide_uri;
    public String model_id;
    public Map<String, Object> tissue;
    public Map<String, Object> tiling;
    public Map<String, Object> inference;

    public ResultLookupRequest(String slideUri, String modelId) {
        this.slide_uri = slideUri;
        this.model_id = modelId;
        this.tissue = new HashMap<>();
        this.tiling = new HashMap<>();
        this.inference = new HashMap<>();
    }

    public ResultLookupRequest(
            String slideUri,
            String modelId,
            Map<String, Object> tissue,
            Map<String, Object> tiling,
            Map<String, Object> inference
    ) {
        this.slide_uri = slideUri;
        this.model_id = modelId;
        this.tissue = tissue != null ? tissue : new HashMap<>();
        this.tiling = tiling != null ? tiling : new HashMap<>();
        this.inference = inference != null ? inference : new HashMap<>();
    }
}
