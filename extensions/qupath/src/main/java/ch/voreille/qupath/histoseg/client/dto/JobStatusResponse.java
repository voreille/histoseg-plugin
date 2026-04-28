package ch.voreille.qupath.histoseg.client.dto;

import java.util.ArrayList;
import java.util.List;

/**
 * Response from GET /jobs/{job_id}.
 */
public class JobStatusResponse {

    public long job_id;
    public String status;
    public List<TaskStatus> tasks = new ArrayList<>();

    public JobStatusResponse() {
    }

    public boolean isTerminal() {
        return "completed".equalsIgnoreCase(status)
                || "failed".equalsIgnoreCase(status)
                || "partial".equalsIgnoreCase(status);
    }
}
