# Appendix Captions (New Visuals and Tables)

## Figures

### Figure A1: Monthly ASP Impact Heatmap
Heatmap of monthly anomaly outcomes after neuro-symbolic fusion in 2022, split into ASP-confirmed and ASP-vetoed counts for Price, Load, Solar, and Wind. The panel contrast quantifies how symbolic constraints filter neural detections over time and across targets.

### Figure A2: Hourly Threshold Uncertainty Bands
Hourly distribution of calibrated anomaly thresholds ($\tau$) shown as 10-90% uncertainty bands with median profiles for Solar, Wind, Load, and Price. The figure highlights context-dependent variability in threshold calibration and reflects weather- and regime-sensitive uncertainty patterns across the day.

### Figure A3: Joint Anomaly UpSet Analysis
UpSet-style intersection analysis of multi-target anomaly co-occurrence, complemented by a pairwise Jaccard overlap matrix. The plot reveals dominant co-anomaly structures (for example, Price with Load or Solar) and summarizes cross-target coupling in detected events.

## Tables

### Table A1: Veto Reason Summary
Top veto categories with counts and percentages of vetoed neural false positives removed by the ASP layer. This table provides an interpretable diagnostic of symbolic filtering behavior, including physically motivated and context-based veto patterns.

### Table A2: Seasonal Performance Summary
Season-wise performance breakdown (Winter, Spring, Summer, Autumn) reporting RMSE and an F1 proxy per target, plus macro seasonal aggregates. The table supports discussion of non-stationarity and seasonal shifts in predictive and detection agreement behavior.

### Table A3: Edge Efficiency Snapshot
Comparative edge-focused benchmark between dCeNN-ELM and LSTM, including inference latency, parameter count, and latency per 1k parameters. This snapshot emphasizes deployment readiness by isolating speed-efficiency trade-offs under resource-constrained inference settings.

## Optional Short Captions (Space-Constrained Layout)

- Figure A1: Monthly ASP-confirmed versus ASP-vetoed anomaly counts by target (2022).
- Figure A2: Hourly calibrated threshold uncertainty bands (10-90%) and median $\tau$ by target.
- Figure A3: Joint target anomaly intersections and pairwise overlap structure.
- Table A1: Top ASP veto categories and share of neural false positives removed.
- Table A2: Seasonal RMSE and F1-proxy breakdown by target and macro average.
- Table A3: Edge-readiness comparison of dCeNN-ELM versus LSTM using latency-per-parameter.
