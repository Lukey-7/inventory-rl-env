Mean episode return over 30 held-out seeds (higher is better; return = minus total cost over 100 days), ± 95% CI. PPO is averaged over its training seeds.

| Config | Random | (s, S) tuned | PPO | PPO per training seed | PPO − (s, S), paired |
|---|---|---|---|---|---|
| easy | -34280.6 ± 1231.3 | -1541.5 ± 21.8 (s=3, S=9) | -1557.4 ± 21.2 | -1568, -1559, -1545 | -15.9 ± 12.2 |
| medium | -33242.5 ± 1202.6 | -1741.0 ± 28.3 (s=12, S=18) | -1737.6 ± 28.6 | -1749, -1730, -1733 | 3.4 ± 14.7 |
| hard | -33870.2 ± 1176.1 | -2442.4 ± 51.5 (s=17, S=29) | -2360.9 ± 44.1 | -2346, -2337, -2399 | 81.5 ± 26.4 |
