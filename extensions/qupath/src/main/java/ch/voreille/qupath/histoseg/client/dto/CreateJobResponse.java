package ch.voreille.qupath.histoseg.client.dto;

/**
 * Response from POST /jobs.
 *
 * Expected backend shape:
 * {
 *   "job_id": 1,
 *   "status": "pending"
 * }
 */
public class CreateJobResponse {

    public long job_id;
    public String status;

    public CreateJobResponse() {
    }
}
