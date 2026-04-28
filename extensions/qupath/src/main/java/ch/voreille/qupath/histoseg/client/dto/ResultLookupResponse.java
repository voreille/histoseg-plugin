package ch.voreille.qupath.histoseg.client.dto;

/**
 * Planned response from POST /results/lookup.
 *
 * Suggested backend shape:
 * {
 *   "found": true,
 *   "result_id": 12,
 *   "status": "completed"
 * }
 */
public class ResultLookupResponse {

    public boolean found;
    public Long result_id;
    public String status;
    public String message;

    public ResultLookupResponse() {
    }

    public boolean hasResult() {
        return found && result_id != null;
    }
}
