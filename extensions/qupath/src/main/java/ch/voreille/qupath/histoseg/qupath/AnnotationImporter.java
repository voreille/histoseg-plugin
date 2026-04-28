package ch.voreille.qupath.histoseg.qupath;

import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;

import java.util.List;

public class AnnotationImporter {

    public static void addObjectsToHierarchy(ImageData<?> imageData, List<PathObject> objects) {
        if (objects == null || objects.isEmpty()) return;

        var hierarchy = imageData.getHierarchy();
        hierarchy.addObjects(objects);
        hierarchy.fireHierarchyChangedEvent(AnnotationImporter.class);
    }
}