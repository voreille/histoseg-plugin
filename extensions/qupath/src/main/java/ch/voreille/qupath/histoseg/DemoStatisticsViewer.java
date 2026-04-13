package ch.voreille.qupath.histoseg;

import java.util.ArrayList;
import java.util.List;

import com.google.gson.JsonObject;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.text.Font;
import javafx.stage.Stage;

public class DemoStatisticsViewer {

    private static final String[] PATTERN_ORDER = {
            "cribriform",
            "micropapillary",
            "solid",
            "papillary",
            "acinar",
            "lepidic"
    };

    private DemoStatisticsViewer() {
    }

    public static void show(JsonObject stats) {
        Stage stage = new Stage();
        stage.setTitle("HistoSeg demo statistics");

        GridPane grid = new GridPane();
        grid.setHgap(12);
        grid.setVgap(12);
        grid.setPadding(new Insets(12));

        grid.add(buildChartPane(
                "Global patterns",
                getObject(stats, "patterns"),
                getInt(stats, "head_b_foreground_area_px", 0),
                "Relative to head B non-background"), 0, 0);

        JsonObject compartments = getObject(stats, "compartments");

        grid.add(buildCompartmentPane(compartments, "Tumor epithelium"), 1, 0);
        grid.add(buildCompartmentPane(compartments, "Stroma"), 0, 1);
        grid.add(buildCompartmentPane(compartments, "Reactive epithelium"), 1, 1);

        Scene scene = new Scene(grid, 1300, 900);
        stage.setScene(scene);
        stage.show();
    }

    private static StackPane buildCompartmentPane(JsonObject compartments, String compartmentName) {
        if (compartments == null || !compartments.has(compartmentName)
                || !compartments.get(compartmentName).isJsonObject()) {
            return buildPlaceholderPane(compartmentName, "No statistics available");
        }

        JsonObject compartment = compartments.getAsJsonObject(compartmentName);
        JsonObject patterns = getObject(compartment, "patterns");
        int areaPx = getInt(compartment, "area_px", 0);

        return buildChartPane(
                compartmentName,
                patterns,
                areaPx,
                "Relative to " + compartmentName + " inside head B non-background");
    }

    private static StackPane buildChartPane(
            String title,
            JsonObject patternStats,
            int denominatorAreaPx,
            String subtitle) {
        if (patternStats == null || patternStats.entrySet().isEmpty()) {
            return buildPlaceholderPane(title, "No statistics available");
        }

        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Ratio (%)");
        yAxis.setForceZeroInRange(true);

        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setAnimated(false);
        chart.setLegendVisible(false);
        chart.setCategoryGap(18);
        chart.setBarGap(4);
        chart.setTitle(title + "\n" + subtitle + "\nArea: " + formatInt(denominatorAreaPx) + " px");

        XYChart.Series<String, Number> argmaxSeries = new XYChart.Series<>();
        List<PatternPoint> points = new ArrayList<>();

        for (String patternName : PATTERN_ORDER) {
            JsonObject p = getPatternStats(patternStats, patternName);
            if (p == null) {
                continue;
            }

            double safe = getDouble(p, "safe_ratio", 0.0) * 100.0;
            double argmax = getDouble(p, "argmax_ratio", 0.0) * 100.0;
            double max = getDouble(p, "max_ratio", 0.0) * 100.0;

            XYChart.Data<String, Number> data = new XYChart.Data<>(patternName, argmax);
            argmaxSeries.getData().add(data);
            points.add(new PatternPoint(patternName, data, safe, argmax, max));
        }

        if (argmaxSeries.getData().isEmpty()) {
            return buildPlaceholderPane(title, "No patterns available");
        }

        chart.getData().add(argmaxSeries);

        Pane overlay = new Pane();
        overlay.setMouseTransparent(true);

        StackPane stack = new StackPane(chart, overlay);
        stack.setPadding(new Insets(6));
        stack.setPrefSize(620, 380);

        Runnable redraw = () -> drawIntervals(chart, yAxis, overlay, points);

        stack.widthProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        stack.heightProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        chart.layoutBoundsProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        chart.needsLayoutProperty().addListener((obs, oldV, newV) -> {
            if (!newV) {
                Platform.runLater(redraw);
            }
        });

        Platform.runLater(redraw);
        return stack;
    }

    private static double valueToPlotY(
            double value,
            double lower,
            double upper,
            double plotMinY,
            double plotMaxY,
            double plotHeight) {
        double clamped = Math.max(lower, Math.min(upper, value));
        double t = (clamped - lower) / (upper - lower);

        // JavaFX chart Y grows downward, so higher values are closer to plotMinY
        return plotMaxY - t * plotHeight;
    }

    private static void drawIntervals(
            BarChart<String, Number> chart,
            NumberAxis yAxis,
            Pane overlay,
            List<PatternPoint> points) {
        overlay.getChildren().clear();

        var plotBackground = chart.lookup(".chart-plot-background");
        if (plotBackground == null) {
            return;
        }

        var plotBoundsScene = plotBackground.localToScene(plotBackground.getBoundsInLocal());
        if (plotBoundsScene == null) {
            return;
        }

        var plotBounds = overlay.sceneToLocal(plotBoundsScene);
        double plotMinY = plotBounds.getMinY();
        double plotMaxY = plotBounds.getMaxY();
        double plotHeight = plotBounds.getHeight();

        double lower = yAxis.getLowerBound();
        double upper = yAxis.getUpperBound();
        if (upper <= lower || plotHeight <= 0) {
            return;
        }

        for (PatternPoint point : points) {
            if (point.data.getNode() == null) {
                continue;
            }

            var barBoundsScene = point.data.getNode().localToScene(point.data.getNode().getBoundsInLocal());
            if (barBoundsScene == null) {
                continue;
            }

            var barBounds = overlay.sceneToLocal(barBoundsScene);
            double x = (barBounds.getMinX() + barBounds.getMaxX()) / 2.0;

            double ySafe = valueToPlotY(point.safePct, lower, upper, plotMinY, plotMaxY, plotHeight);
            double yArgmax = valueToPlotY(point.argmaxPct, lower, upper, plotMinY, plotMaxY, plotHeight);
            double yMax = valueToPlotY(point.maxPct, lower, upper, plotMinY, plotMaxY, plotHeight);

            Line vertical = new Line(x, yMax, x, ySafe);
            vertical.setStrokeWidth(2.0);
            vertical.setStroke(Color.BLACK);

            Line capTop = new Line(x - 6, yMax, x + 6, yMax);
            capTop.setStrokeWidth(2.0);
            capTop.setStroke(Color.BLACK);

            Line capBottom = new Line(x - 6, ySafe, x + 6, ySafe);
            capBottom.setStrokeWidth(2.0);
            capBottom.setStroke(Color.BLACK);

            Line centerTick = new Line(x - 5, yArgmax, x + 5, yArgmax);
            centerTick.setStrokeWidth(2.0);
            centerTick.setStroke(Color.DARKRED);

            overlay.getChildren().addAll(vertical, capTop, capBottom, centerTick);
        }
    }

    private static StackPane buildPlaceholderPane(String title, String message) {
        Label label = new Label(title + "\n" + message);
        label.setFont(Font.font(16));
        StackPane pane = new StackPane(label);
        pane.setPadding(new Insets(12));
        pane.setPrefSize(620, 380);
        pane.setStyle("-fx-border-color: lightgray; -fx-border-width: 1; -fx-background-color: white;");
        return pane;
    }

    private static JsonObject getPatternStats(JsonObject patterns, String patternName) {
        if (patterns == null || !patterns.has(patternName) || !patterns.get(patternName).isJsonObject()) {
            return null;
        }
        return patterns.getAsJsonObject(patternName);
    }

    private static JsonObject getObject(JsonObject parent, String key) {
        if (parent == null || !parent.has(key) || !parent.get(key).isJsonObject()) {
            return null;
        }
        return parent.getAsJsonObject(key);
    }

    private static int getInt(JsonObject obj, String key, int fallback) {
        if (obj == null || !obj.has(key) || obj.get(key).isJsonNull()) {
            return fallback;
        }
        try {
            return obj.get(key).getAsInt();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static double getDouble(JsonObject obj, String key, double fallback) {
        if (obj == null || !obj.has(key) || obj.get(key).isJsonNull()) {
            return fallback;
        }
        try {
            return obj.get(key).getAsDouble();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String formatInt(int value) {
        return String.format("%,d", value);
    }

    private static class PatternPoint {
        final String name;
        final XYChart.Data<String, Number> data;
        final double safePct;
        final double argmaxPct;
        final double maxPct;

        PatternPoint(
                String name,
                XYChart.Data<String, Number> data,
                double safePct,
                double argmaxPct,
                double maxPct) {
            this.name = name;
            this.data = data;
            this.safePct = safePct;
            this.argmaxPct = argmaxPct;
            this.maxPct = maxPct;
        }
    }
}