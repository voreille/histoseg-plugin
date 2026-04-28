package ch.voreille.qupath.histoseg.client;

import ch.voreille.qupath.histoseg.client.dto.*;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class HistoSegClient {

    private final String serverUrl;
    private final HttpClient httpClient;
    private final Gson gson = new Gson();

    public HistoSegClient(String serverUrl) {
        this.serverUrl = stripTrailingSlash(serverUrl);
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    public CreateJobResponse submitJob(CreateJobRequest request) throws Exception {
        String body = postJson("/jobs", gson.toJson(request));
        return gson.fromJson(body, CreateJobResponse.class);
    }

    public JobStatusResponse getJob(long jobId) throws Exception {
        String body = getJson("/jobs/" + jobId);
        return gson.fromJson(body, JobStatusResponse.class);
    }

    public QueueStatusResponse getQueueStatus() throws Exception {
        String body = getJson("/queue");
        return gson.fromJson(body, QueueStatusResponse.class);
    }

    public QueueStatusResponse pauseQueue() throws Exception {
        String body = postJson("/queue/pause", "{}");
        return gson.fromJson(body, QueueStatusResponse.class);
    }

    public QueueStatusResponse resumeQueue() throws Exception {
        String body = postJson("/queue/resume", "{}");
        return gson.fromJson(body, QueueStatusResponse.class);
    }

    public ResultLookupResponse lookupResult(ResultLookupRequest request) throws Exception {
        String body = postJson("/results/lookup", gson.toJson(request));
        return gson.fromJson(body, ResultLookupResponse.class);
    }

    public ResultResponse getResult(long resultId) throws Exception {
        String body = getJson("/results/" + resultId);
        return gson.fromJson(body, ResultResponse.class);
    }

    // Keep these raw for developer/debug sync routes.
    public JsonObject runSynchronousTissueSegmentation(JsonObject payload) throws Exception {
        String body = postJson("/segment/tissue", payload.toString());
        return JsonParser.parseString(body).getAsJsonObject();
    }

    public JsonObject runSynchronousWSISegmentation(JsonObject payload) throws Exception {
        String body = postJson("/segment/wsi", payload.toString());
        return JsonParser.parseString(body).getAsJsonObject();
    }

    private String getJson(String endpoint) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + endpoint))
                .timeout(Duration.ofMinutes(10))
                .header("Accept", "application/json")
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return checkResponse(response);
        } catch (HistoSegClientException e) {
            throw e;
        } catch (Exception e) {
            throw new HistoSegClientException("Failed to call " + endpoint, e);
        }
    }

    private String postJson(String endpoint, String jsonBody) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + endpoint))
                .timeout(Duration.ofMinutes(10))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return checkResponse(response);
        } catch (HistoSegClientException e) {
            throw e;
        } catch (Exception e) {
            throw new HistoSegClientException("Failed to call " + endpoint, e);
        }
    }

    private static String checkResponse(HttpResponse<String> response) {
        int statusCode = response.statusCode();
        String body = response.body();

        if (statusCode < 200 || statusCode >= 300) {
            throw new HistoSegClientException(statusCode, "Server error " + statusCode, body);
        }

        return body;
    }

    private static String stripTrailingSlash(String url) {
        if (url == null) {
            throw new IllegalArgumentException("Server URL cannot be null");
        }

        String trimmed = url.trim();

        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }

        return trimmed;
    }
}