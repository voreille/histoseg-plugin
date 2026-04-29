package ch.voreille.qupath.histoseg;

import ch.voreille.qupath.histoseg.commands.HistoSegCommands;
import javafx.scene.control.MenuItem;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;

public class HistoSegExtension implements QuPathExtension {

    @Override
    public void installExtension(QuPathGUI qupath) {
        HistoSegCommands commands = new HistoSegCommands(qupath);

        var menu = qupath.getMenu("Extensions>HistoSeg", true);

        menu.getItems().add(createMenuItem(
                "Submit current slide...",
                commands::submitCurrentSlide
        ));

        menu.getItems().add(createMenuItem(
                "Submit all project slides...",
                commands::submitAllProjectSlides
        ));

        menu.getItems().add(createMenuItem(
                "Import existing result for current slide...",
                commands::importResultForCurrentSlide
        ));

        menu.getItems().add(createMenuItem(
                "Queue status...",
                commands::showQueueStatus
        ));

        // Optional: keep old synchronous/debug endpoints during development.
        menu.getItems().add(createMenuItem(
                "Developer: synchronous tissue segmentation...",
                commands::runSynchronousTissueSegmentation
        ));

        menu.getItems().add(createMenuItem(
                "Developer: synchronous WSI segmentation...",
                commands::runSynchronousWSISegmentation
        ));
    }

    @Override
    public String getName() {
        return "HistoSeg";
    }

    @Override
    public String getDescription() {
        return "Submit HistoSeg WSI segmentation jobs, check queue status, and import GeoJSON annotations into QuPath.";
    }

    private static MenuItem createMenuItem(String text, Runnable action) {
        MenuItem item = new MenuItem(text);
        item.setOnAction(e -> action.run());
        return item;
    }
}