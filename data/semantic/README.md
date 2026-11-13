# Frozen category-prediction data

Files contain 10,859 training requirements, 1,245 validation requirements, 1,245 test requirements and 1,245 category rows. Hashes are in checksums.json.

Each requirement has query, categories, original_id and level. The original_id field is the unique category ID. A category row has category_id, category_name, category_label (full path) and category_level. Names alone may be ambiguous.

D-Acc accepts the reference ID or a descendant on the same path. See docs/provenance.md for split and synthetic-augmentation provenance.
