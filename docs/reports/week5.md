# Weekly report: week 5

Add xoshiro256pp shuffle with lemires bias reduction techniques,
and splitmix64 to initialize state generators.
Implement idx file loading and conversion from big-endian to little-endian.
Add aligned allocators for vectors to optimize cache loading.
Add dockerfile for convenience, so the project can be tested on older systems
with less-modern build dependencies.

## Time log

| Date | Time used | Notes |
| ----- | ------------- | ------ |
| 28.9. | 5h | Implement xoshiro256++, splitmix64 |
| 29.9. | 5h | Finish shuffle+testing, lemired bias reduction |
| 1.10. | 7h | Implement idx file loading with error checking and conversion |
| 2.10  | 3h | Add dockerfile for convenience |
| total | 20h | |

