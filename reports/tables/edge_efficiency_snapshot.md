# Edge Efficiency Snapshot

| model | parameters | inf_latency_ms | latency_ms_per_1k_params | avg_rmse | rmse_solar | rmse_wind | rmse_load | rmse_price | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM (Baseline-Sequential) | 79364 | 0.328408 | 0.004138 | 1629.695127 | 0.08567 | 0.226257 | 6266.1333 | 252.33528 |  |
| dCeNN-ELM (Proposed) | 8016 | 0.087301 | 0.010891 | 1221.041723 | 0.102424 | 0.323572 | 4663.765121 | 219.975775 |  |
| Edge Readiness Gain (dCeNN vs LSTM) |  |  | 0.379952 |  |  |  |  |  | Absolute latency speedup: 3.76x (lower is better) |

Notes: latency_ms_per_1k_params = inf_latency_ms / (parameters / 1000). Lower is better.
