package ch.voreille.qupath.histoseg.qupath;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.projects.ProjectImageEntry;

import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

public class QuPathSlideUtils {

    public static ImageData<?> getCurrentImageData(QuPathGUI qupath) {
        var viewer = qupath.getViewer();
        return viewer != null ? viewer.getImageData() : null;
    }

    public static String getSlideUri(ImageData<?> imageData) {
        var uris = imageData.getServer().getURIs();
        return getPrimaryUri(uris);
    }

    public static List<String> getProjectSlideUris(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null)
            return List.of();

        return project.getImageList().stream()
                .map(QuPathSlideUtils::safeGetPrimaryUri)
                .filter(Objects::nonNull)
                .toList();
    }

    private static String safeGetPrimaryUri(ProjectImageEntry<?> entry) {
        try {
            return getPrimaryUri(entry.getURIs());
        } catch (IOException e) {
            return null;
        }
    }

    private static String getPrimaryUri(Collection<URI> uris) {
        if (uris == null || uris.isEmpty())
            return null;

        for (URI uri : uris) {
            if ("file".equalsIgnoreCase(uri.getScheme()))
                return uri.toString();
        }

        return uris.iterator().next().toString();
    }
}