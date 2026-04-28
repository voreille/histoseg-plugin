package ch.voreille.qupath.histoseg.client.dto;

/**
 * Response from GET /queue or a future GET /queue/summary.
 *
 * The current backend may only return:
 * {
 *   "paused": false
 * }
 *
 * Extra fields are optional for future queue summaries.
 */
public class QueueStatusResponse {

    public boolean paused;

    public Integer pending;
    public Integer running;
    public Integer completed;
    public Integer failed;

    public QueueStatusResponse() {
    }
}
