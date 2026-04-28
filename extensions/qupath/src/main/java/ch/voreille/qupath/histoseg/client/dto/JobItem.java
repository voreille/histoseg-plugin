package ch.voreille.qupath.histoseg.client.dto;

import java.util.HashMap;
import java.util.Map;

/**
 * One slide/model item submitted to the HistoSeg backend.
 *
 * Mirrors the backend JobItem Pydantic model:
 * {
 *   "slide_uri": "...",
 *   "model_id": "default",
 *   "tissue": {},
 *   "tiling": {},
 *   "inference": {}
 * }
 */
public class JobItem {

    public String slide_uri;
    public String model_id;
    public Map<String, Object> tissue;
    public Map<String, Object> tiling;
    public Map<String, Object> inference;

    public JobItem(String slideUri, String modelId) {
        this.slide_uri = slideUri;
        this.model_id = modelId;
        this.tissue = new HashMap<>();
        this.tiling = new HashMap<>();
        this.inference = new HashMap<>();
    }

    public JobItem(
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
