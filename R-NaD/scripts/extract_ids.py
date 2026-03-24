import os
import re
import json

def slugify(txt):
    # CamelCase to snake_case transition:
    # This regex is a simplified version of MegaCrit.Sts2.Core.Helpers.StringHelper.Slugify
    # It adds an underscore before any capital letter preceded by another character.
    # Pattern: ([A-Za-z0-9])([A-Z])
    text = re.sub(r'([A-Za-z0-9])([A-Z])', r'\1_\2', txt.strip())
    # Convert to uppercase
    text = text.upper()
    # Replace whitespace with underscore
    text = re.sub(r'\s+', '_', text)
    # Remove non-alphanumeric (excluding underscore)
    text = re.sub(r'[^A-Z0-9_]', '', text)
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text)
    return text.strip('_')

def extract_ids(base_path, model_dir):
    full_path = os.path.join(base_path, model_dir)
    if not os.path.isdir(full_path):
        print(f"Directory not found: {full_path}")
        return []
    
    ids = []
    for filename in sorted(os.listdir(full_path)):
        if filename.endswith(".cs"):
            class_name = filename[:-3]
            game_id = slugify(class_name)
            ids.append(game_id)
    return ids

def main():
    base_path = "/home/ubuntu/src/R-NaD-StS2/StS2_Decompiled"
    
    mapping = {
        "CARDS": "MegaCrit.Sts2.Core.Models.Cards",
        "MONSTERS": "MegaCrit.Sts2.Core.Models.Monsters",
        "RELICS": "MegaCrit.Sts2.Core.Models.Relics",
        "POTIONS": "MegaCrit.Sts2.Core.Models.Potions",
        "POWERS": "MegaCrit.Sts2.Core.Models.Powers",
        "EVENTS": "MegaCrit.Sts2.Core.Models.Events",
        "ORBS": "MegaCrit.Sts2.Core.Models.Orbs",
    }
    
    all_ids = {}
    for key, model_dir in mapping.items():
        print(f"Extracting {key} from {model_dir}...")
        ids = extract_ids(base_path, model_dir)
        all_ids[key] = ids
        print(f"Found {len(ids)} IDs.")

    output_file = "/home/ubuntu/src/R-NaD-StS2/R-NaD/scripts/game_ids.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_ids, f, indent=4)
    
    print(f"\nDone! IDs saved to {output_file}")

if __name__ == "__main__":
    main()
