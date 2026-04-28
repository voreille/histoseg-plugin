package ch.voreille.qupath.histoseg.client.dto;

/**
 * One task entry returned inside GET /jobs/{job_id}.
 */
public class TaskStatus {

    public long task_id;
    public String status;
    public String slide_path;
    public String model_id;
    public Double progress;
    public String stage;
    public String error_message;
    public Long result_id;

    public TaskStatus() {
    }

    public boolean hasResult() {
        return result_id != null;
    }

    public boolean isCompleted() {
        return "completed".equalsIgnoreCase(status) || "cached".equalsIgnoreCase(status);
    }

    public boolean isFailed() {
        return "failed".equalsIgnoreCase(status);
    }
}
