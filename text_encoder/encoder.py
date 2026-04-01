import json
import re
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def clean_text(text):
    if not text:
        return ""
    # Strip HTML-like tags like [gold], [blue], [/gold], etc.
    text = re.sub(r'\[/?\w+\]', '', text)
    # Strip hex color tags like [#ff0000], etc.
    text = re.sub(r'\[#\w+\]', '', text)
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Strip extra spaces
    text = ' '.join(text.split())
    return text

def normalize_id(id_str):
    if not id_str:
        return ""
    # Strip upgrade suffix (+), trim, and normalize to UPPER_CASE
    nid = str(id_str).split('+')[0].strip().upper().replace(" ", "_").replace("-", "_")
    # Strip punctuation
    for char in "!?.(),":
        nid = nid.replace(char, "")
    return nid

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Store by type
    type_entities = {
        "CARDS": {},
        "RELICS": {},
        "POWERS": {},
        "MONSTERS": {}
    }
    
    # Process Cards
    cards_path = os.path.join(script_dir, 'cards.json')
    with open(cards_path, 'r') as f:
        cards = json.load(f)
        for card in cards:
            cid = normalize_id(card.get('id'))
            desc = card.get('description') or card.get('name', "")
            text = clean_text(desc)
            if cid:
                type_entities["CARDS"][cid] = text

    # Process Relics
    relics_path = os.path.join(script_dir, 'relics.json')
    if os.path.exists(relics_path):
        with open(relics_path, 'r') as f:
            relics = json.load(f)
            for relic in relics:
                rid = normalize_id(relic.get('id'))
                desc = relic.get('description', "")
                text = clean_text(desc)
                if rid:
                    type_entities["RELICS"][rid] = text

    # Process Powers
    powers_path = os.path.join(script_dir, 'powers.json')
    if os.path.exists(powers_path):
        with open(powers_path, 'r') as f:
            powers = json.load(f)
            for power in powers:
                pid = normalize_id(power.get('id'))
                desc = power.get('description', "")
                text = clean_text(desc)
                if pid:
                    type_entities["POWERS"][pid] = text

    # Process Monsters
    monsters_path = os.path.join(script_dir, 'monsters.json')
    if os.path.exists(monsters_path):
        with open(monsters_path, 'r') as f:
            monsters = json.load(f)
            for monster in monsters:
                mid = normalize_id(monster.get('id'))
                name = monster.get('name', "")
                text = clean_text(name)
                if mid:
                    type_entities["MONSTERS"][mid] = text

    # Flatten for encoding
    all_types = list(type_entities.keys())
    all_ids = []
    all_texts = []
    id_to_type = {}
    
    for t in all_types:
        for eid, text in type_entities[t].items():
            all_ids.append(eid)
            all_texts.append(text)
            id_to_type[len(all_ids)-1] = t
    
    print(f"Encoding {len(all_texts)} entities...")
    embeddings = model.encode(all_texts)
    
    results = {t: {} for t in all_types}
    for i, emb in enumerate(embeddings):
        t = id_to_type[i]
        eid = all_ids[i]
        results[t][eid] = emb
    
    # Save as PKL
    output_path = os.path.join(script_dir, 'embeddings.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved split embeddings to embeddings.pkl")
    for t in all_types:
        print(f"  {t}: {len(results[t])} items")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
