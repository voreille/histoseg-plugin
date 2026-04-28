package ch.voreille.qupath.histoseg.client.dto;

import java.util.List;

/**
 * Request body for POST /jobs.
 */
public class CreateJobRequest {

    public List<JobItem> items;

    public CreateJobRequest(List<JobItem> items) {
        this.items = items;
    }
}
