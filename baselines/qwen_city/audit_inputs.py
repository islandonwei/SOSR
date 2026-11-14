import hashlib, json
from pathlib import Path
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()
