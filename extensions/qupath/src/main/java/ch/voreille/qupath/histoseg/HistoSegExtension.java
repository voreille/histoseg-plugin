package ch.voreille.qupath.histoseg;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
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
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

public class HistoSegExtension implements QuPathExtension {

    private static final String DEFAULT_SERVER = "http://localhost:8090";

    private static final String TISSUE_ENDPOINT = "/segment/tissue";
    private static final String WSI_SEGMENTATION_ENDPOINT = "/segment/wsi";

    @Override
    public void installExtension(QuPathGUI qupath) {
        var menu = qupath.getMenu("Extensions>HistoSeg", true);

        MenuItem tissueItem = new MenuItem("Tissue segmentation (FastAPI)...");
        tissueItem.setOnAction(e -> runTissueSegmentation(qupath));
        menu.getItems().add(tissueItem);

        MenuItem wsiItem = new MenuItem("WSI segmentation (FastAPI)...");
        wsiItem.setOnAction(e -> runWSISegmentation(qupath));
        menu.getItems().add(wsiItem);
    }

    @Override
    public String getName() {
        return "HistoSeg";
    }

    @Override
    public String getDescription() {
        return "Call a FastAPI server to compute tissue and WSI segmentation and import as annotations.";
    }

    private static void showError(String title, String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(title);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private static Optional<String> askServerUrl() {
        TextInputDialog dlg = new TextInputDialog(DEFAULT_SERVER);
        dlg.setTitle("HistoSeg");
        dlg.setHeaderText("FastAPI server URL (use SSH tunnel to localhost)");
        dlg.setContentText("Server URL:");
        return dlg.showAndWait()
                .map(String::trim)
                .map(url -> url.endsWith("/") ? url.substring(0, url.length() - 1) : url);
    }

    private void runTissueSegmentation(QuPathGUI qupath) {
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

        Optional<String> serverUrlOpt = askServerUrl();
        if (serverUrlOpt.isEmpty()) {
            return;
        }

        String serverUrl = serverUrlOpt.get();
        JsonObject payload = buildTissuePayload(slideUri);

        new Thread(() -> {
            try {
                String responseBody = postJson(serverUrl + TISSUE_ENDPOINT, payload.toString());
                List<PathObject> objects = featureCollectionStringToAnnotations(responseBody, "Tissue");

                Platform.runLater(() -> {
                    var hierarchy = imageData.getHierarchy();
                    hierarchy.addObjects(objects);
                    hierarchy.fireHierarchyChangedEvent(this);
                });
            } catch (Exception ex) {
                Platform.runLater(() -> showError("HistoSeg", ex.toString()));
            }
        }, "HistoSeg-Tissue").start();
    }

    private void runWSISegmentation(QuPathGUI qupath) {
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

        Optional<String> serverUrlOpt = askServerUrl();
        if (serverUrlOpt.isEmpty()) {
            return;
        }

        String serverUrl = serverUrlOpt.get();
        JsonObject payload = buildWSISegmentationPayload(slideUri);

        new Thread(() -> {
            try {
                String responseBody = postJson(serverUrl + WSI_SEGMENTATION_ENDPOINT, payload.toString());
                List<PathObject> objects = parseWSISegmentationResponse(responseBody);

                Platform.runLater(() -> {
                    var hierarchy = imageData.getHierarchy();
                    hierarchy.addObjects(objects);
                    hierarchy.fireHierarchyChangedEvent(this);
                });
            } catch (Exception ex) {
                Platform.runLater(() -> showError("HistoSeg", ex.toString()));
            }
        }, "HistoSeg-WSI").start();
    }

    private static JsonObject buildTissuePayload(String slideUri) {
        JsonObject payload = new JsonObject();
        payload.addProperty("slide_uri", slideUri);
        payload.addProperty("seg_level", -1);

        payload.addProperty("sthresh", 20);
        payload.addProperty("sthresh_up", 255);
        payload.addProperty("mthresh", 7);
        payload.addProperty("close", 0);
        payload.addProperty("use_otsu", false);

        JsonObject filterParams = new JsonObject();
        filterParams.addProperty("a_t", 100);
        filterParams.addProperty("a_h", 16);
        filterParams.addProperty("max_n_holes", 10);
        payload.add("filter_params", filterParams);

        payload.addProperty("ref_patch_size", 512);
        payload.addProperty("min_area_px_level0", 0);
        payload.addProperty("simplify_tol_px_level0", 0.0);

        return payload;
    }

    private static JsonObject buildWSISegmentationPayload(String slideUri) {
        JsonObject payload = new JsonObject();

        payload.addProperty("slide_uri", slideUri);
        payload.addProperty("seg_level", -1);

        payload.addProperty("sthresh", 20);
        payload.addProperty("sthresh_up", 255);
        payload.addProperty("mthresh", 7);
        payload.addProperty("close", 0);
        payload.addProperty("use_otsu", false);

        JsonObject filterParams = new JsonObject();
        filterParams.addProperty("a_t", 100);
        filterParams.addProperty("a_h", 16);
        filterParams.addProperty("max_n_holes", 10);
        payload.add("filter_params", filterParams);

        payload.addProperty("ref_patch_size", 512);
        payload.addProperty("min_area_px_level0", 0);
        payload.addProperty("simplify_tol_px_level0", 0.0);

        payload.addProperty("patch_level", 0);
        payload.addProperty("patch_size", 896);
        payload.addProperty("step_size", 896);

        payload.addProperty("contour_fn", "four_pt");
        payload.addProperty("center_shift", 0.5);
        payload.addProperty("use_padding", true);
        payload.addProperty("max_workers", 4);

        payload.addProperty("output_target_mpp", 4.0);
        payload.addProperty("batch_size", 8);
        payload.addProperty("num_workers", 0);

        return payload;
    }

    private static String getPrimaryUri(Collection<URI> uris) {
        if (uris == null || uris.isEmpty()) {
            return null;
        }

        for (URI u : uris) {
            if ("file".equalsIgnoreCase(u.getScheme())) {
                return u.toString();
            }
        }
        return uris.iterator().next().toString();
    }

    private static String postJson(String url, String jsonBody) throws Exception {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
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

    private static List<PathObject> parseWSISegmentationResponse(String json) {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        List<PathObject> objects = new ArrayList<>();

        if (root.has("tissue") && root.get("tissue").isJsonObject()) {
            JsonObject tissueFc = root.getAsJsonObject("tissue");
            objects.addAll(featureCollectionObjectToAnnotations(tissueFc, "Tissue"));
        }

        if (root.has("outputs") && root.get("outputs").isJsonObject()) {
            JsonObject outputs = root.getAsJsonObject("outputs");
            for (Map.Entry<String, JsonElement> entry : outputs.entrySet()) {
                if (!entry.getValue().isJsonObject()) {
                    continue;
                }
                JsonObject fc = entry.getValue().getAsJsonObject();
                objects.addAll(featureCollectionObjectToAnnotations(fc, null));
            }
        }

        return objects;
    }

    private static List<PathObject> featureCollectionStringToAnnotations(String geojson, String fallbackClassName) {
        JsonObject root = JsonParser.parseString(geojson).getAsJsonObject();
        return featureCollectionObjectToAnnotations(root, fallbackClassName);
    }

    private static List<PathObject> featureCollectionObjectToAnnotations(JsonObject featureCollection,
            String fallbackClassName) {
        List<PathObject> objects = new ArrayList<>();

        if (!featureCollection.has("features") || !featureCollection.get("features").isJsonArray()) {
            return objects;
        }

        JsonArray features = featureCollection.getAsJsonArray("features");
        ImagePlane plane = ImagePlane.getDefaultPlane();

        for (JsonElement featEl : features) {
            if (!featEl.isJsonObject()) {
                continue;
            }

            JsonObject feat = featEl.getAsJsonObject();
            if (!feat.has("geometry") || !feat.get("geometry").isJsonObject()) {
                continue;
            }

            JsonObject geom = feat.getAsJsonObject("geometry");
            String type = geom.has("type") ? geom.get("type").getAsString() : "";
            String className = extractClassName(feat, fallbackClassName);

            if ("Polygon".equalsIgnoreCase(type)) {
                objects.addAll(polygonGeometryToAnnotations(geom, className, plane));
            } else if ("MultiPolygon".equalsIgnoreCase(type)) {
                objects.addAll(multiPolygonGeometryToAnnotations(geom, className, plane));
            }
        }

        return objects;
    }

    private static String extractClassName(JsonObject feature, String fallbackClassName) {
        String className = fallbackClassName;
        if (feature.has("properties") && feature.get("properties").isJsonObject()) {
            JsonObject props = feature.getAsJsonObject("properties");
            if (props.has("class") && !props.get("class").isJsonNull()) {
                className = props.get("class").getAsString();
            }
        }
        return className;
    }

    private static List<PathObject> polygonGeometryToAnnotations(JsonObject geom, String className, ImagePlane plane) {
        List<PathObject> objects = new ArrayList<>();

        if (!geom.has("coordinates") || !geom.get("coordinates").isJsonArray()) {
            return objects;
        }

        JsonArray rings = geom.getAsJsonArray("coordinates");
        ROI roi = polygonRingsToROI(rings, plane);
        if (roi == null) {
            return objects;
        }

        objects.add(createAnnotationObject(roi, className));
        return objects;
    }

    private static List<PathObject> multiPolygonGeometryToAnnotations(JsonObject geom, String className,
            ImagePlane plane) {
        List<PathObject> objects = new ArrayList<>();

        if (!geom.has("coordinates") || !geom.get("coordinates").isJsonArray()) {
            return objects;
        }

        JsonArray polygons = geom.getAsJsonArray("coordinates");
        for (JsonElement polyEl : polygons) {
            if (!polyEl.isJsonArray()) {
                continue;
            }

            ROI roi = polygonRingsToROI(polyEl.getAsJsonArray(), plane);
            if (roi == null) {
                continue;
            }

            objects.add(createAnnotationObject(roi, className));
        }

        return objects;
    }

    private static ROI polygonRingsToROI(JsonArray rings, ImagePlane plane) {
        if (rings.size() == 0) {
            return null;
        }

        JsonElement outerEl = rings.get(0);
        if (!outerEl.isJsonArray()) {
            return null;
        }

        List<Point2> outerPoints = ringToPoints(outerEl.getAsJsonArray());
        if (outerPoints.size() < 3) {
            return null;
        }

        // QuPath 0.7 ROIs helper does not support this holes overload here.
        // For now, ignore inner rings and keep only the outer boundary.
        return ROIs.createPolygonROI(outerPoints, plane);
    }

    private static List<Point2> ringToPoints(JsonArray ring) {
        List<Point2> pts = new ArrayList<>();

        for (JsonElement pEl : ring) {
            if (!pEl.isJsonArray()) {
                continue;
            }

            JsonArray p = pEl.getAsJsonArray();
            if (p.size() < 2) {
                continue;
            }

            double x = p.get(0).getAsDouble();
            double y = p.get(1).getAsDouble();
            pts.add(new Point2(x, y));
        }

        if (!pts.isEmpty() && isClosedRingDuplicate(pts)) {
            pts.remove(pts.size() - 1);
        }

        return pts;
    }

    private static boolean isClosedRingDuplicate(List<Point2> pts) {
        if (pts.size() < 2) {
            return false;
        }
        Point2 first = pts.get(0);
        Point2 last = pts.get(pts.size() - 1);
        return first.getX() == last.getX() && first.getY() == last.getY();
    }

    private static PathObject createAnnotationObject(ROI roi, String className) {
        if (className != null && !className.isBlank()) {
            PathClass pathClass = PathClass.fromString(className);
            return PathObjects.createAnnotationObject(roi, pathClass);
        }
        return PathObjects.createAnnotationObject(roi);
    }
}