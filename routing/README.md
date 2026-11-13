# FSSP and routing baselines

This module evaluates routing over fixed category candidate sets. It contains FSSP, an RNII-guided ROSE implementation, the existing Candidate-A* implementation labelled DROMC in the archives, and layered multi-source Dijkstra.

Run all commands from the repository root.

## Requirements

- Linux or WSL
- A C++17 compiler, such as g++
- Python 3.11 or later for benchmark control and statistics

## Compile and check

```bash
python routing/build.py
```

The command builds `routing/build/sosr_route` and runs the small-graph correctness, timeout, and geometric-bound checks. The checks use generated fixtures.

A CMake entry point is also provided:

```bash
cmake -S routing -B routing/build
cmake --build routing/build -j2
ctest --test-dir routing/build --output-on-failure
```

## Prepare road data

Extract `SOSR-road-data-20261113.zip` at the repository root. Each dataset directory contains its graph, coordinates, category mapping, POI membership, and H2H order files.

```bash
python scripts/check_data.py --roads data/roads
```

See the [road input guide](../data/roads/README.md) for filenames and the [data manifest](../data/roads/manifest.json) for hashes.

## Run queries

Run a diagnostic subset:

```bash
python routing/benchmark.py --data-root data/roads --datasets NYC --limit 1 --output runs/routing_check
```

Index construction is performed for the chosen network even when a single query is selected.

Run the full benchmark:

```bash
python routing/benchmark.py --data-root data/roads --output runs/routing_1800
```

The benchmark contains 1,800 fixed queries and three paired repetitions across four methods, producing 21,600 measurements. The optional `--cpu` argument selects one CPU from the process's permitted affinity.

Inputs are listed in `routing/QUERIES.json` and `routing/inputs/`. Each line in a dataset's query file contains:

```text
query_id source_vertex target_vertex number_of_requirements category_1 ... category_k
```

Query vertex IDs are zero-based. Road-file vertex IDs are one-based, and the loader converts them. Category IDs refer to that network's `Categories_ID.txt`.

## Outputs

The output directory contains:

- `r<repeat>_<dataset>/measurements.jsonl` — per-query results;
- `RESULTS.csv` and `RESULTS.json` — aggregate statistics;
- `RUN_SCOPE.json` — whether the full fixed manifest or a subset was run;
- `data/` and `work/` — input links and generated index files.

Each successful result contains the distance and ordered service vertices. Routes may revisit vertices, and one vertex may serve consecutive requirements, including at the source or target.

## Measurement protocol

Reported latency includes query-related initialization, planning, and service-vertex reconstruction. It excludes HSM inference, index setup, full road-walk recovery, fork/IPC, and result verification.

Queries have a 30-second budget. Timeouts are retained as censored observations. The statistics use nearest-rank median/P95 over each query's median across three repetitions.

The FSSP implementation calibrates geometric lower bounds against graph edge weights. The historical DROMC label refers to the retained Candidate-A* code; the implementation identities and full measurement protocol are detailed in [docs/routing.md](../docs/routing.md).

The saved comparison and historical manuscript plots are described in the [results guide](../results/README.md).
