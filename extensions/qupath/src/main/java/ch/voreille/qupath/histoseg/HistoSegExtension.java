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

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LinearRing;
import org.locationtech.jts.geom.MultiPolygon;
import org.locationtech.jts.geom.Polygon;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextInputDialog;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.interfaces.ROI;

public class HistoSegExtension implements QuPathExtension {

    private static final String DEFAULT_SERVER = "http://localhost:8090";

    private static final String TISSUE_ENDPOINT = "/segment/tissue";
    private static final String WSI_SEGMENTATION_ENDPOINT = "/segment/wsi";

    private static final String PROP_CLASS = "class";
    private static final String PROP_OBJECT_NAME = "object_name";
    private static final String PROP_HEAD_DISPLAY_NAME = "head_display_name";

    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory();

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

    private void runTissueSegmentation(QuPathGUI qupath) {
        ImageData<?> imageData = getCurrentImageData(qupath);
        if (imageData == null) {
            return;
        }

        String slideUri = getSlideUri(imageData);
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
                List<PathObject> objects = featureCollectionStringToAnnotations(responseBody, "Tissue", "Tissue");
                Platform.runLater(() -> addObjectsToHierarchy(imageData, objects));
            } catch (Exception ex) {
                Platform.runLater(() -> showError("HistoSeg", ex.toString()));
            }
        }, "HistoSeg-Tissue").start();
    }

    private void runWSISegmentation(QuPathGUI qupath) {
        ImageData<?> imageData = getCurrentImageData(qupath);
        if (imageData == null) {
            return;
        }

        String slideUri = getSlideUri(imageData);
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
                ParsedWSIResponse parsed = parseWSISegmentationResponse(responseBody);

                Platform.runLater(() -> {
                    addObjectsToHierarchy(imageData, parsed.objects);
                    if (parsed.statisticsJson != null) {
                        DemoStatisticsViewer.show(parsed.statisticsJson);
                    }
                });
            } catch (Exception ex) {
                Platform.runLater(() -> showError("HistoSeg", ex.toString()));
            }
        }, "HistoSeg-WSI").start();
    }

    private static ImageData<?> getCurrentImageData(QuPathGUI qupath) {
        var viewer = qupath.getViewer();
        if (viewer == null || viewer.getImageData() == null) {
            showError("HistoSeg", "No image open.");
            return null;
        }
        return viewer.getImageData();
    }

    private static String getSlideUri(ImageData<?> imageData) {
        ImageServer<?> server = imageData.getServer();
        return getPrimaryUri(server.getURIs());
    }

    private static class ParsedWSIResponse {
        final List<PathObject> objects;
        final JsonObject statisticsJson;

        ParsedWSIResponse(List<PathObject> objects, JsonObject statisticsJson) {
            this.objects = objects;
            this.statisticsJson = statisticsJson;
        }
    }

    private static void addObjectsToHierarchy(ImageData<?> imageData, List<PathObject> objects) {
        if (objects == null || objects.isEmpty()) {
            return;
        }

        var hierarchy = imageData.getHierarchy();
        hierarchy.addObjects(objects);
        hierarchy.fireHierarchyChangedEvent(HistoSegExtension.class);
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
                .filter(s -> !s.isBlank())
                .map(HistoSegExtension::stripTrailingSlash);
    }

    private static String stripTrailingSlash(String url) {
        return url.endsWith("/") ? url.substring(0, url.length() - 1) : url;
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
        JsonObject payload = buildTissuePayload(slideUri);

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

        for (URI uri : uris) {
            if ("file".equalsIgnoreCase(uri.getScheme())) {
                return uri.toString();
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

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new RuntimeException("Server error " + response.statusCode() + ": " + response.body());
        }

        return response.body();
    }

    private static ParsedWSIResponse parseWSISegmentationResponse(String json) {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        List<PathObject> objects = new ArrayList<>();

        JsonObject tissue = getObject(root, "tissue");
        if (tissue != null) {
            objects.addAll(featureCollectionObjectToAnnotations(tissue, "Tissue", "Tissue"));
        }

        JsonObject outputs = getObject(root, "outputs");
        if (outputs != null) {
            for (Map.Entry<String, JsonElement> entry : outputs.entrySet()) {
                if (!entry.getValue().isJsonObject()) {
                    continue;
                }
                objects.addAll(featureCollectionObjectToAnnotations(entry.getValue().getAsJsonObject(), null, null));
            }
        }

        JsonObject statisticsJson = getObject(root, "statistics");
        return new ParsedWSIResponse(objects, statisticsJson);
    }

    private static List<PathObject> featureCollectionStringToAnnotations(
            String geojson,
            String fallbackClassName,
            String fallbackObjectName
    ) {
        JsonObject root = JsonParser.parseString(geojson).getAsJsonObject();
        return featureCollectionObjectToAnnotations(root, fallbackClassName, fallbackObjectName);
    }

    private static List<PathObject> featureCollectionObjectToAnnotations(
            JsonObject featureCollection,
            String fallbackClassName,
            String fallbackObjectName
    ) {
        List<PathObject> objects = new ArrayList<>();

        JsonArray features = getArray(featureCollection, "features");
        if (features == null) {
            return objects;
        }

        ImagePlane plane = ImagePlane.getDefaultPlane();

        for (JsonElement featureElement : features) {
            if (!featureElement.isJsonObject()) {
                continue;
            }

            JsonObject feature = featureElement.getAsJsonObject();
            JsonObject geometryObject = getObject(feature, "geometry");
            if (geometryObject == null) {
                continue;
            }

            Geometry geometry = geometryFromGeoJson(geometryObject);
            if (geometry == null || geometry.isEmpty()) {
                continue;
            }

            ROI roi = GeometryTools.geometryToROI(geometry, plane);
            if (roi == null || roi.isEmpty()) {
                continue;
            }

            String className = extractClassName(feature, fallbackClassName);
            String objectName = extractObjectName(feature, fallbackObjectName);

            objects.add(createAnnotationObject(roi, className, objectName));
        }

        return objects;
    }

    private static JsonObject getObject(JsonObject parent, String key) {
        if (parent == null || !parent.has(key) || !parent.get(key).isJsonObject()) {
            return null;
        }
        return parent.getAsJsonObject(key);
    }

    private static JsonArray getArray(JsonObject parent, String key) {
        if (parent == null || !parent.has(key) || !parent.get(key).isJsonArray()) {
            return null;
        }
        return parent.getAsJsonArray(key);
    }

    private static String getString(JsonObject parent, String key) {
        if (parent == null || !parent.has(key) || parent.get(key).isJsonNull()) {
            return null;
        }

        String value = parent.get(key).getAsString();
        if (value == null || value.isBlank()) {
            return null;
        }

        return value;
    }

    private static String extractClassName(JsonObject feature, String fallbackClassName) {
        JsonObject properties = getObject(feature, "properties");
        String className = getString(properties, PROP_CLASS);
        return className != null ? className : fallbackClassName;
    }

    private static String extractObjectName(JsonObject feature, String fallbackObjectName) {
        JsonObject properties = getObject(feature, "properties");

        String objectName = getString(properties, PROP_OBJECT_NAME);
        if (objectName != null) {
            return objectName;
        }

        String headDisplayName = getString(properties, PROP_HEAD_DISPLAY_NAME);
        if (headDisplayName != null) {
            return headDisplayName;
        }

        return fallbackObjectName;
    }

    private static Geometry geometryFromGeoJson(JsonObject geom) {
        if (geom == null || !geom.has("type")) {
            return null;
        }

        String type = geom.get("type").getAsString();
        if (!geom.has("coordinates") || !geom.get("coordinates").isJsonArray()) {
            return null;
        }

        JsonArray coordinates = geom.getAsJsonArray("coordinates");

        try {
            return switch (type) {
                case "Polygon" -> polygonFromCoordinates(coordinates);
                case "MultiPolygon" -> multiPolygonFromCoordinates(coordinates);
                default -> null;
            };
        } catch (Exception e) {
            return null;
        }
    }

    private static Geometry polygonFromCoordinates(JsonArray ringsArray) {
        if (ringsArray == null || ringsArray.size() == 0) {
            return null;
        }

        LinearRing shell = linearRingFromJsonArray(getJsonArray(ringsArray, 0));
        if (shell == null) {
            return null;
        }

        LinearRing[] holes = new LinearRing[Math.max(0, ringsArray.size() - 1)];
        for (int i = 1; i < ringsArray.size(); i++) {
            LinearRing hole = linearRingFromJsonArray(getJsonArray(ringsArray, i));
            if (hole == null) {
                return null;
            }
            holes[i - 1] = hole;
        }

        Polygon polygon = GEOMETRY_FACTORY.createPolygon(shell, holes);
        return makeGeometryValid(polygon);
    }

    private static Geometry multiPolygonFromCoordinates(JsonArray polygonsArray) {
        if (polygonsArray == null || polygonsArray.size() == 0) {
            return null;
        }

        List<Polygon> polygons = new ArrayList<>();

        for (JsonElement polygonElement : polygonsArray) {
            if (!polygonElement.isJsonArray()) {
                continue;
            }

            Geometry geometry = polygonFromCoordinates(polygonElement.getAsJsonArray());
            if (geometry == null || geometry.isEmpty()) {
                continue;
            }

            if (geometry instanceof Polygon polygon) {
                polygons.add(polygon);
            } else if (geometry instanceof MultiPolygon multiPolygon) {
                for (int i = 0; i < multiPolygon.getNumGeometries(); i++) {
                    Geometry g = multiPolygon.getGeometryN(i);
                    if (g instanceof Polygon polygon) {
                        polygons.add(polygon);
                    }
                }
            }
        }

        if (polygons.isEmpty()) {
            return null;
        }

        MultiPolygon multiPolygon = GEOMETRY_FACTORY.createMultiPolygon(polygons.toArray(new Polygon[0]));
        return makeGeometryValid(multiPolygon);
    }

    private static Geometry makeGeometryValid(Geometry geometry) {
        if (geometry == null) {
            return null;
        }

        if (geometry.isValid()) {
            return geometry;
        }

        Geometry repaired = geometry.buffer(0);
        if (repaired == null || repaired.isEmpty()) {
            return null;
        }

        return repaired;
    }

    private static LinearRing linearRingFromJsonArray(JsonArray ringArray) {
        Coordinate[] coordinates = coordinatesFromRing(ringArray);
        if (coordinates == null) {
            return null;
        }

        try {
            return GEOMETRY_FACTORY.createLinearRing(coordinates);
        } catch (Exception e) {
            return null;
        }
    }

    private static Coordinate[] coordinatesFromRing(JsonArray ringArray) {
        if (ringArray == null || ringArray.size() < 4) {
            return null;
        }

        List<Coordinate> coordinates = new ArrayList<>();

        for (JsonElement pointElement : ringArray) {
            if (!pointElement.isJsonArray()) {
                continue;
            }

            JsonArray point = pointElement.getAsJsonArray();
            if (point.size() < 2) {
                continue;
            }

            double x = point.get(0).getAsDouble();
            double y = point.get(1).getAsDouble();
            coordinates.add(new Coordinate(x, y));
        }

        if (coordinates.size() < 3) {
            return null;
        }

        Coordinate first = coordinates.get(0);
        Coordinate last = coordinates.get(coordinates.size() - 1);

        if (!sameCoordinate(first, last)) {
            coordinates.add(new Coordinate(first.x, first.y));
        }

        if (coordinates.size() < 4) {
            return null;
        }

        return coordinates.toArray(new Coordinate[0]);
    }

    private static boolean sameCoordinate(Coordinate a, Coordinate b) {
        return a != null && b != null && a.x == b.x && a.y == b.y;
    }

    private static JsonArray getJsonArray(JsonArray parent, int index) {
        if (parent == null || index < 0 || index >= parent.size()) {
            return null;
        }

        JsonElement element = parent.get(index);
        return element != null && element.isJsonArray() ? element.getAsJsonArray() : null;
    }

    private static PathObject createAnnotationObject(ROI roi, String className, String objectName) {
        PathObject object;

        if (className != null && !className.isBlank()) {
            PathClass pathClass = PathClass.fromString(className);
            object = PathObjects.createAnnotationObject(roi, pathClass);
        } else {
            object = PathObjects.createAnnotationObject(roi);
        }

        if (objectName != null && !objectName.isBlank()) {
            object.setName(objectName);
        }

        return object;
    }
}