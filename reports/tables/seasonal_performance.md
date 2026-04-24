# Seasonal Performance

| season | target | n_points | rmse | f1_proxy | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Winter | Load | 2087 | 2035.457509 | 0.487069 | 113 | 217 | 21 |
| Winter | Macro Avg | 8348 | 549.486455 | 0.507052 | 939 | 1520 | 84 |
| Winter | Price | 2087 | 162.167642 | 0.751152 | 489 | 294 | 30 |
| Winter | Solar | 2087 | 0.121235 | 0.268041 | 117 | 627 | 12 |
| Winter | Wind | 2087 | 0.199436 | 0.521945 | 220 | 382 | 21 |
| Spring | Load | 2208 | 1977.801994 | 0.405063 | 80 | 223 | 12 |
| Spring | Macro Avg | 8832 | 533.66167 | 0.422108 | 1115 | 2367 | 137 |
| Spring | Price | 2208 | 156.47124 | 0.569146 | 463 | 652 | 49 |
| Spring | Solar | 2208 | 0.123985 | 0.068441 | 36 | 935 | 45 |
| Spring | Wind | 2208 | 0.249462 | 0.645783 | 536 | 557 | 31 |
| Summer | Load | 2208 | 1712.838291 | 0.301158 | 39 | 168 | 13 |
| Summer | Macro Avg | 8832 | 498.606993 | 0.486781 | 2092 | 2476 | 127 |
| Summer | Price | 2208 | 281.142593 | 0.784776 | 1165 | 624 | 15 |
| Summer | Solar | 2208 | 0.136491 | 0.183372 | 118 | 994 | 57 |
| Summer | Wind | 2208 | 0.310598 | 0.677817 | 770 | 690 | 42 |
| Autumn | Load | 2184 | 1709.937026 | 0.328889 | 37 | 137 | 14 |
| Autumn | Macro Avg | 8736 | 480.407579 | 0.497776 | 1478 | 1631 | 99 |
| Autumn | Price | 2184 | 211.324366 | 0.767339 | 780 | 436 | 37 |
| Autumn | Solar | 2184 | 0.113143 | 0.19571 | 73 | 582 | 18 |
| Autumn | Wind | 2184 | 0.255781 | 0.699168 | 588 | 476 | 30 |

Notes: f1_proxy is the seasonal agreement score between neural anomaly flags and ASP-confirmed anomalies.
