Mean episode return over 30 held-out seeds (higher is better; return = minus total cost over 100 days), ± 95% CI. ★ = strongest baseline, chosen on the tuning seeds. The last column is paired (same demand on each seed).

| Config | Optimal (DP) | (s, S) | weekday (s, S) | capped base-stock | PPO (3 seeds) | PPO per seed | PPO − strongest baseline |
|---|---|---|---|---|---|---|---|
| easy | -1537.7 ± 23.2 | -1541.5 ± 21.8 | -1572.1 ± 24.5 ★ | -1669.9 ± 24.2 | -1572.4 ± 21.0 | -1562, -1573, -1582 | -0.3 ± 17.7 |
| medium | -1683.5 ± 25.2 | -1741.0 ± 28.3 | -1740.3 ± 27.0 ★ | -1814.9 ± 24.6 | -1743.4 ± 29.7 | -1764, -1726, -1740 | -3.1 ± 18.8 |
| hard | not tractable | -2442.4 ± 51.5 | -2329.9 ± 41.2 ★ | -3463.8 ± 42.1 | -2342.8 ± 44.4 | -2337, -2347, -2344 | -12.9 ± 23.0 |

Extra cost compared with the optimal policy (paired, ± 95% CI); lower is better:

| Config | (s, S) | weekday (s, S) | capped base-stock | PPO (3 seeds) |
|---|---|---|---|---|
| easy | 0.2% ± 0.2 | 2.2% ± 1.1 | 8.6% ± 1.0 | 2.3% ± 1.0 |
| medium | 3.4% ± 1.0 | 3.4% ± 1.1 | 7.8% ± 1.0 | 3.6% ± 1.1 |

Random ordering, for scale: easy -34,281, medium -33,242, hard -33,870.
