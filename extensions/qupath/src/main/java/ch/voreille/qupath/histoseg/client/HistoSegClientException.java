package ch.voreille.qupath.histoseg.client;

public class HistoSegClientException extends RuntimeException {

    private final int statusCode;
    private final String responseBody;

    public HistoSegClientException(int statusCode, String message, String responseBody) {
        super(message);
        this.statusCode = statusCode;
        this.responseBody = responseBody;
    }

    public HistoSegClientException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = -1;
        this.responseBody = null;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getResponseBody() {
        return responseBody;
    }
}