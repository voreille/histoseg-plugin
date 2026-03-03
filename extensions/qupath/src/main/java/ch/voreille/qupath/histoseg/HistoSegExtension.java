package ch.voreille.qupath.histoseg;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextInputDialog;
import qupath.lib.geom.Point2;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;


public class HistoSegExtension implements QuPathExtension {

    private static final String DEFAULT_SERVER = "http://localhost:8000";
    private static final String ENDPOINT = "/tissue/contours";

    @Override
    public void installExtension(QuPathGUI qupath) {
        var menu = qupath.getMenu("Extensions>HistoSeg", true);

        MenuItem item = new MenuItem("Tissue contours (FastAPI)...");
        item.setOnAction(e -> runTissueContours(qupath));
        menu.getItems().add(item);
    }

    @Override
    public String getName() {
        return "HistoSeg";
    }

    @Override
    public String getDescription() {
        return "Call a FastAPI server to compute tissue contours and import as annotations.";
    }

    private static void showError(String title, String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(title);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void runTissueContours(QuPathGUI qupath) {

        var viewer = qupath.getViewer();
        if (viewer == null || viewer.getImageData() == null) {
            showError("HistoSeg", "No image open.");
            return;
        }
        var imageData = viewer.getImageData();

        ImageServer<?> server = imageData.getServer();
        String slideUri = getPrimaryUri(server.getURIs());
        if (slideUri == null) {
            showError("HistoSeg", "Could not determine slide URI.");
            return;
        }

        TextInputDialog dlg = new TextInputDialog(DEFAULT_SERVER);
        dlg.setTitle("HistoSeg");
        dlg.setHeaderText("FastAPI server URL (use SSH tunnel to localhost)");
        dlg.setContentText("Server URL:");
        Optional<String> serverUrlOpt = dlg.showAndWait();
        if (serverUrlOpt.isEmpty())
            return;

        String serverUrl = serverUrlOpt.get().trim();
        if (serverUrl.endsWith("/"))
            serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

        JsonObject payload = new JsonObject();
        payload.addProperty("slide_uri", slideUri);
        payload.addProperty("seg_level", -1);

        String finalServerUrl = serverUrl;

        new Thread(() -> {
            try {
                String responseBody = postJson(finalServerUrl + ENDPOINT, payload.toString());
                List<PathObject> objects = geoJsonToAnnotations(responseBody);

                Platform.runLater(() -> {
                    var hierarchy = imageData.getHierarchy();
                    hierarchy.addObjects(objects);
                    hierarchy.fireHierarchyChangedEvent(this);
                });

            } catch (Exception ex) {
                Platform.runLater(() -> showError("HistoSeg", ex.toString()));
            }
        }, "HistoSeg-FastAPI").start();
    }

    private static String getPrimaryUri(Collection<URI> uris) {
        if (uris == null || uris.isEmpty())
            return null;

        for (URI u : uris) {
            if ("file".equalsIgnoreCase(u.getScheme()))
                return u.toString();
        }
        return uris.iterator().next().toString();
    }

    private static String postJson(String url, String jsonBody) throws Exception {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)   // <-- important
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(10))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        HttpResponse<String> resp = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
            throw new RuntimeException("Server error " + resp.statusCode() + ": " + resp.body());
        }
        return resp.body();
    }

    private static List<PathObject> geoJsonToAnnotations(String geojson) {

        JsonObject root = JsonParser.parseString(geojson).getAsJsonObject();
        JsonArray features = root.getAsJsonArray("features");

        var pathClass = PathClassFactory.getPathClass("Tissue");
        var plane = ImagePlane.getDefaultPlane();

        List<PathObject> objects = new ArrayList<>();

        for (JsonElement featEl : features) {
            JsonObject feat = featEl.getAsJsonObject();
            JsonObject geom = feat.getAsJsonObject("geometry");
            String type = geom.get("type").getAsString();

            if (!"Polygon".equalsIgnoreCase(type))
                continue;

            JsonArray rings = geom.getAsJsonArray("coordinates");
            if (rings.size() == 0)
                continue;

            JsonArray outer = rings.get(0).getAsJsonArray();

            List<Point2> pts = new ArrayList<>();
            for (JsonElement pEl : outer) {
                JsonArray p = pEl.getAsJsonArray();
                double x = p.get(0).getAsDouble();
                double y = p.get(1).getAsDouble();
                pts.add(new Point2(x, y));
            }

            ROI roi = ROIs.createPolygonROI(pts, plane);
            PathObject obj = PathObjects.createAnnotationObject(roi, pathClass);
            objects.add(obj);
        }

        return objects;
    }
}