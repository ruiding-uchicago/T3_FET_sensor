#!/usr/bin/env python3
"""
Anonymize JSON files:
1. Remove DOI and Original_Text fields
2. Rename files from DOI to SHA256 hash (consistent mapping)
3. Apply to both descriptor_injected_jsons and convert_csv_to_json/json_output_raw
4. Save DOI→hash mapping to JSON file
"""

import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Directories to process
DIRS_TO_PROCESS = [
    BASE_DIR / "descriptor_injected_jsons",
    BASE_DIR / "convert_csv_to_json" / "json_output_raw",
]

# Fields to remove from JSON content
FIELDS_TO_REMOVE = ["DOI", "Original_Text"]

# Mapping output file
MAPPING_FILE = BASE_DIR / "doi_to_hash_mapping.json"

def doi_to_hash(doi: str) -> str:
    """Convert DOI to a consistent 12-char hash."""
    return hashlib.sha256(doi.encode()).hexdigest()[:12]

def process_directory(dir_path: Path, doi_hash_map: dict):
    """Process all JSON files in a directory."""
    if not dir_path.exists():
        print(f"⚠️  Directory not found: {dir_path}")
        return

    json_files = list(dir_path.glob("*.json"))
    print(f"\n📁 Processing {dir_path.name}/ ({len(json_files)} files)")

    for json_path in json_files:
        # 统一用文件名作为DOI来源
        doi_from_filename = json_path.stem.replace("_", "/")
        file_hash = doi_to_hash(doi_from_filename)

        # 记录映射
        doi_hash_map[doi_from_filename] = file_hash

        # Read JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error reading {json_path.name}: {e}")
            continue

        # Remove sensitive fields
        for field in FIELDS_TO_REMOVE:
            data.pop(field, None)

        # Write to new filename
        new_path = dir_path / f"{file_hash}.json"
        with open(new_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Delete old file
        if json_path != new_path:
            json_path.unlink()

    print(f"  ✅ Processed {len(json_files)} files")

def main():
    print("=" * 60)
    print("Anonymizing JSON files")
    print("=" * 60)

    doi_hash_map = {}

    for dir_path in DIRS_TO_PROCESS:
        process_directory(dir_path, doi_hash_map)

    # Save mapping
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(doi_hash_map, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Mapping saved: {MAPPING_FILE.name} ({len(doi_hash_map)} DOIs)")
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
