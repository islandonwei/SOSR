# Routing protocol

The driver reads paired arcs as an undirected graph. Input vertices are one-based, while query manifests and outputs are zero-based. Integer edge weights are divided by 10. Routes may revisit vertices. One vertex can satisfy consecutive requirements, including at the source or target.

## Methods

- FSSP uses H2H distances, representative upper bounds, calibrated grid lower bounds, SPF chains and DP. Representative count 100 is a target with tolerance 10 and a final-grid fallback. SPF uses 4 candidate draws and up to 2 retained roots per category.
- ROSE is an independent RNII-guided implementation with 50  seed42 farthest-first landmarks, exact refinement and query-local caching.
- DROMC preserves the existing candidate-layer A* implementation. It is not a verified full reproduction of the cited DROMC algorithm.
- LayeredDijkstra initializes each graph propagation with the previous candidate layer's DP labels and reads the next layer.

Geometric calibration scans all edges and verifies that scaled coordinate distances do not exceed edge weights. Conservative floating-point bounds are used. The legacy unscaled projection had confirmed overestimation cases. The release entry point uses calibration.

## Fixed comparison

The manifest has 1,200 length3 queries across NYC/BAY/COL/FLA and LLL/MMM/HHH, plus 600 length1/5 queries on NYC/FLA. Endpoints cover five distance bins. Candidate/endpoint selection was frozen before timing. HSM is outside this experiment.

Methods return distance and ordered service vertices. Timing includes candidate processing, query initialization, pruning/search and service-vertex reconstruction. It excludes index setup, fork/IPC, output serialization, verification and full road-walk recovery. Cleanup of retained global/cache state after return is outside timing for all methods.

Each dataset/repetition uses a fresh loaded-index parent; each query runs in a fresh child. Method order rotates. Three paired repetitions give 21,600 measurements.

The deadline is 30 seconds. Timeouts store planner_ms=null and censor_ms=30000. Median/P95 are nearest-rank quantiles over per-query medians of three repetitions, propagating right censoring. A null aggregate quantile means above budget. Errors stop execution and are not timeouts.

Saved raw files are in results/routing/raw. Independent reference distances are in routing/references. The result script recomputes the summaries. The comparison completed on 2026-11-01 and is supplemental to earlier manuscript figures.

Compilation uses -O3 -fno-fast-math -ffp-contract=off. The build script runs 420 small graphs across four methods and a timeout fixture. Full workloads need the separate matching data asset. New latency depends on hardware and load.
