# Weekly report: week 5

Add xoshiro256pp shuffle with lemires bias reduction techniques,
and splitmix64 to initialize state generators.
Implement idx file loading and conversion from big-endian to little-endian.
Add aligned allocators for vectors to optimize cache loading.

## Time log

| Date | Time used | Notes |
| ----- | ------------- | ------ |
| 28.9. | 5h | Implement xoshiro256++, splitmix64 |
| 29.9. | 5h | Finish shuffle+testing, lemired bias reduction |
| 2.10. | 7h | Implement idx file loading with error checking and conversion |
| total | 17h | |

