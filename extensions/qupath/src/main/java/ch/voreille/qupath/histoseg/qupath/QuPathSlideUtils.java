package ch.voreille.qupath.histoseg.qupath;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;

import java.net.URI;
import java.util.Collection;
import java.util.List;

public class QuPathSlideUtils {

    public static ImageData<?> getCurrentImageData(QuPathGUI qupath) {
        var viewer = qupath.getViewer();
        return viewer != null ? viewer.getImageData() : null;
    }

    public static String getSlideUri(ImageData<?> imageData) {
        var uris = imageData.getServer().getURIs();
        return getPrimaryUri(uris);
    }

    public static List<URI> getSelectedProjectSlideUris(QuPathGUI qupath) {
        return List.of(); // TODO
    }

    private static String getPrimaryUri(Collection<URI> uris) {
        for (URI uri : uris) {
            if ("file".equalsIgnoreCase(uri.getScheme()))
                return uri.toString();
        }
        return uris.iterator().next().toString();
    }
}