package ch.voreille.qupath.histoseg.ui;

import ch.voreille.qupath.histoseg.client.dto.QueueStatusResponse;
import javafx.scene.control.Alert;

public class QueueStatusDialog {

    public static void show(QueueStatusResponse status) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setContentText("Queue paused: " + status.paused);
        alert.showAndWait();
    }
}