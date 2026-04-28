package ch.voreille.qupath.histoseg.ui;

import javafx.scene.control.TextInputDialog;

public class ServerUrlDialog {

    public static String ask() {
        TextInputDialog dlg = new TextInputDialog("http://localhost:8000");
        return dlg.showAndWait().orElse(null);
    }
}