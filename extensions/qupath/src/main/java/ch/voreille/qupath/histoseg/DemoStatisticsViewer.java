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
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
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

        VBox globalPanel = buildGlobalPanel(stats);

        StackPane centeredTop = new StackPane(globalPanel);
        centeredTop.setPrefWidth(1500); // match scene width
        centeredTop.setPadding(new Insets(0, 0, 10, 0));

        grid.add(centeredTop, 0, 0, 2, 1);

        JsonObject compartments = getObject(stats, "compartments");
        grid.add(buildCompartmentPanel(compartments, "Tumor epithelium"), 0, 1);
        grid.add(buildCompartmentPanel(compartments, "Stroma"), 1, 1);

        Scene scene = new Scene(grid, 1500, 900);
        stage.setScene(scene);
        stage.show();
    }

    private static VBox buildGlobalPanel(JsonObject stats) {
        JsonObject patterns = getObject(stats, "patterns");
        double areaUm2 = getDouble(stats, "head_b_foreground_area_um2", 0.0);

        return buildChartPanel(
                "Relative Area of Predicted Patterns",
                patterns,
                areaUm2);
    }

    private static VBox buildCompartmentPanel(JsonObject compartments, String compartmentName) {
        if (compartments == null || !compartments.has(compartmentName)
                || !compartments.get(compartmentName).isJsonObject()) {
            return buildPlaceholderPane(compartmentName, "No statistics available");
        }

        JsonObject compartment = compartments.getAsJsonObject(compartmentName);
        JsonObject patterns = getObject(compartment, "patterns");
        double areaUm2 = getDouble(compartment, "area_um2", 0.0);

        return buildChartPanel(
                "Relative Area of Predicted Patterns in " + compartmentName,
                patterns,
                areaUm2);
    }

    private static VBox buildChartPanel(
            String title,
            JsonObject patternStats,
            double denominatorAreaUm2) {
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
        chart.setTitle(
                title
                        + "\nTotal Area: " + formatAreaUm2(denominatorAreaUm2));

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

        StackPane chartPane = new StackPane(chart, overlay);
        chartPane.setPadding(new Insets(6));
        chartPane.setPrefSize(700, 320);

        Runnable redraw = () -> drawIntervals(chart, yAxis, overlay, points);

        chart.layoutBoundsProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        chart.widthProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        chart.heightProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        overlay.widthProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));
        overlay.heightProperty().addListener((obs, oldV, newV) -> Platform.runLater(redraw));

        Platform.runLater(() -> Platform.runLater(redraw));

        TextArea summaryArea = new TextArea(buildPatternSummaryText(points));
        summaryArea.setEditable(false);
        summaryArea.setWrapText(false);
        summaryArea.setPrefRowCount(8);
        summaryArea.setPrefColumnCount(65);
        summaryArea.setStyle("-fx-font-family: monospace;");

        VBox panel = new VBox(8, chartPane, summaryArea);
        panel.setPadding(new Insets(6));
        panel.setPrefSize(720, 420);
        return panel;
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

            double ySafe = valueToPlotY(point.safePct, lower, upper, plotMaxY, plotHeight);
            double yMax = valueToPlotY(point.maxPct, lower, upper, plotMaxY, plotHeight);

            Line vertical = new Line(x, yMax, x, ySafe);
            vertical.setStrokeWidth(1.5);
            vertical.setStroke(Color.BLACK);

            Line capTop = new Line(x - 6, yMax, x + 6, yMax);
            capTop.setStrokeWidth(1.5);
            capTop.setStroke(Color.BLACK);

            Line capBottom = new Line(x - 6, ySafe, x + 6, ySafe);
            capBottom.setStrokeWidth(1.5);
            capBottom.setStroke(Color.BLACK);

            overlay.getChildren().addAll(vertical, capTop, capBottom);
        }
    }

    private static double valueToPlotY(
            double value,
            double lower,
            double upper,
            double plotMaxY,
            double plotHeight) {
        double clamped = Math.max(lower, Math.min(upper, value));
        double t = (clamped - lower) / (upper - lower);
        return plotMaxY - t * plotHeight;
    }

    private static String buildPatternSummaryText(List<PatternPoint> points) {
        StringBuilder sb = new StringBuilder();
        for (PatternPoint p : points) {
            sb.append(String.format(
                    "%-16s: %5.1f [%.1f - %.1f]%%%n",
                    p.name,
                    p.argmaxPct,
                    p.safePct,
                    p.maxPct));
        }
        return sb.toString().trim();
    }

    private static VBox buildPlaceholderPane(String title, String message) {
        Label label = new Label(title + "\n" + message);
        label.setFont(Font.font(16));

        BorderPane pane = new BorderPane(label);
        pane.setPadding(new Insets(12));
        pane.setStyle("-fx-border-color: lightgray; -fx-border-width: 1; -fx-background-color: white;");
        pane.setPrefSize(720, 420);

        VBox wrapper = new VBox(pane);
        wrapper.setPrefSize(720, 420);
        return wrapper;
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

    private static String formatAreaUm2(double value) {
        return String.format("%,.0f µm²", value);
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