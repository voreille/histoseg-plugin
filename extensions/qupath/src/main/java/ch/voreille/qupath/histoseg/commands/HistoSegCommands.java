package ch.voreille.qupath.histoseg.commands;

import ch.voreille.qupath.histoseg.client.HistoSegClient;
import ch.voreille.qupath.histoseg.client.dto.*;
import ch.voreille.qupath.histoseg.qupath.QuPathSlideUtils;
import ch.voreille.qupath.histoseg.qupath.AnnotationImporter;
import ch.voreille.qupath.histoseg.geojson.GeoJsonParser;
import ch.voreille.qupath.histoseg.ui.ServerUrlDialog;
import ch.voreille.qupath.histoseg.ui.QueueStatusDialog;

import com.google.gson.JsonObject;

import javafx.application.Platform;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;

import java.util.List;

public class HistoSegCommands {

    private final QuPathGUI qupath;

    public HistoSegCommands(QuPathGUI qupath) {
        this.qupath = qupath;
    }

    public void submitCurrentSlide() {
        ImageData<?> imageData = QuPathSlideUtils.getCurrentImageData(qupath);
        if (imageData == null) return;

        String slideUri = QuPathSlideUtils.getSlideUri(imageData);
        String server = ServerUrlDialog.ask();
        if (server == null) return;

        HistoSegClient client = new HistoSegClient(server);

        try {
            JobItem item = new JobItem(slideUri, "default", null, null, null);
            CreateJobRequest req = new CreateJobRequest(List.of(item));
            CreateJobResponse resp = client.submitJob(req);
            System.out.println("Submitted job: " + resp.job_id);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void submitSelectedProjectSlides() {
        var uris = QuPathSlideUtils.getSelectedProjectSlideUris(qupath);
        String server = ServerUrlDialog.ask();
        if (server == null) return;

        HistoSegClient client = new HistoSegClient(server);

        try {
            var items = uris.stream()
                .map(uri -> new JobItem(uri.toString(), "default", null, null, null))
                .toList();

            CreateJobRequest req = new CreateJobRequest(items);
            client.submitJob(req);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void importResultForCurrentSlide() {
        ImageData<?> imageData = QuPathSlideUtils.getCurrentImageData(qupath);
        if (imageData == null) return;

        String slideUri = QuPathSlideUtils.getSlideUri(imageData);
        String server = ServerUrlDialog.ask();
        if (server == null) return;

        HistoSegClient client = new HistoSegClient(server);

        new Thread(() -> {
            try {
                ResultLookupRequest req = new ResultLookupRequest(slideUri, "default");
                ResultLookupResponse lookup = client.lookupResult(req);

                if (!lookup.found) return;

                ResultResponse result = client.getResult(lookup.result_id);
                JsonObject geo = result.geojson;

                var objects = GeoJsonParser.parseFeatureCollection(geo);
                Platform.runLater(() -> AnnotationImporter.addObjectsToHierarchy(imageData, objects));

            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    public void showQueueStatus() {
        String server = ServerUrlDialog.ask();
        if (server == null) return;

        HistoSegClient client = new HistoSegClient(server);

        try {
            QueueStatusResponse status = client.getQueueStatus();
            QueueStatusDialog.show(status);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void runSynchronousTissueSegmentation() {}
    public void runSynchronousWSISegmentation() {}
}