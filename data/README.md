# Data

Run all commands from the repository root.

## Semantic requirements

The files in `data/semantic/` are included in the repository.

| File | Contents | Count |
|---|---|---:|
| `train.json` | Training requirements | 10,859 |
| `val.json` | Validation requirements | 1,245 |
| `test.json` | Test requirements | 1,245 |
| `categories.csv` | Category IDs and hierarchy paths | 1,245 |

Each category has one validation and one test requirement. A requirement record has the following form:

```json
{
  "query": "looking for a place to join winter sports",
  "categories": ["Hockey Club"],
  "original_id": "63be6904847c3692a84b9c0c",
  "level": 3
}
```

`original_id` identifies a category in `categories.csv`. Category names can be ambiguous, so evaluation uses unique IDs. The CSV contains `category_id`, `category_level`, `category_name`, and `category_label`, where the last field stores the complete hierarchy path.

Check the frozen files:

```bash
python scripts/check_data.py
```

Checksums are in [semantic/checksums.json](semantic/checksums.json). Dataset origins, augmentation, and the historical split are described in [docs/provenance.md](../docs/provenance.md).

## Road networks

Download [SOSR-road-data-20261113.zip](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-road-data-20261113.zip) for the matching NYC, BAY, COL, and FLA inputs. Extract it at the repository root:

```text
data/roads/
├── NYC/
├── BAY/
├── COL/
├── FLA/
└── manifest.json
```

Each network has the following files:

| Filename | Contents |
|---|---|
| `USA-road-d.<dataset>.gr` | Road graph with paired arcs and integer weights |
| `USA-road-d.<dataset>.co` | Vertex coordinates |
| `order.txt` | Recorded H2H ordering |
| `poi_node_mapping.txt` | POI categories mapped to vertices |
| `Categories_ID.txt` | Network category IDs and names |

Verify them after extraction:

```bash
python scripts/check_data.py --roads data/roads
```

The input graph's paired arcs are loaded as an undirected graph. Vertex IDs in the graph files are one-based; query manifests use zero-based IDs. The loader divides stored edge weights by 10. Routing semantics are documented in [routing/README.md](../routing/README.md).

The release asset contains the full road inputs. Verify it against `data/roads/manifest.json` after extraction.

## Additional model data

Qwen-City training uses the external CityInstruction-London corpus. Exact input hashes and preparation commands are in [baselines/qwen_city/README.md](../baselines/qwen_city/README.md).

Third-party data and model sources are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).
