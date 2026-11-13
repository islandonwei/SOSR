# Road inputs

The companion SOSR-road-data-20261113.zip contains the 20 files described by manifest.json. Extract it at the repository root to populate this directory with NYC/BAY/COL/FLA.

Each network has:

- USA-road-d.DATASET.gr — paired arcs and integer weights;
- USA-road-d.DATASET.co — coordinates;
- order.txt — the recorded H2H ordering;
- poi_node_mapping.txt — supplied POI/category-to-vertex mapping;
- Categories_ID.txt — network category IDs and names.

Run python scripts/check_data.py --roads data/roads after extraction. Original files are linked into a new run directory. Generated index files are written there.

Inputs derive from the DIMACS 9th challenge roads and the project's Foursquare POI mapping. Preserve their source terms when publishing the data asset. 

## Download

[Download the road-data archive](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-road-data-20261113.zip).
