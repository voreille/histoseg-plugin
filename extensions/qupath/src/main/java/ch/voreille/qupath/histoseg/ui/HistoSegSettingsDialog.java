package ch.voreille.qupath.histoseg.ui;

import ch.voreille.qupath.histoseg.settings.HistoSegPreferences;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;

public class HistoSegSettingsDialog {

    public static void show() {
        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("HistoSeg settings");
        dialog.setHeaderText("Configure HistoSeg backend settings");

        ButtonType saveButton = new ButtonType("Save", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(saveButton, ButtonType.CANCEL);

        TextField serverUrlField = new TextField(HistoSegPreferences.getServerUrl());
        TextField modelIdField = new TextField(HistoSegPreferences.getModelId());

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);

        grid.add(new Label("Server URL:"), 0, 0);
        grid.add(serverUrlField, 1, 0);

        grid.add(new Label("Model ID:"), 0, 1);
        grid.add(modelIdField, 1, 1);

        dialog.getDialogPane().setContent(grid);

        dialog.setResultConverter(button -> {
            if (button == saveButton) {
                HistoSegPreferences.setServerUrl(stripTrailingSlash(serverUrlField.getText()));
                HistoSegPreferences.setModelId(modelIdField.getText().trim());
            }
            return null;
        });

        dialog.showAndWait();
    }

    private static String stripTrailingSlash(String url) {
        if (url == null) {
            return "";
        }

        String trimmed = url.trim();

        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }

        return trimmed;
    }
}