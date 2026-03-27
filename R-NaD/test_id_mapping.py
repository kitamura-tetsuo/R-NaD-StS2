import sys
import os

# Mocking the environment for rnad_bridge
CARD_VOCAB = {
    "UNKNOWN": 0,
    "STRIKE_IRONCLAD": 1,
    "ONE_TWO_PUNCH": 359,
}

def get_card_idx_mock(card_id):
    if not card_id:
        return 0
    if isinstance(card_id, dict):
        card_id = card_id.get("id") or card_id.get("name")
    
    # Original normalization
    # cid = str(card_id).split('+')[0].strip().upper().replace(" ", "_")
    
    # Proposed normalization
    cid = str(card_id).split('+')[0].strip().upper().replace(" ", "_").replace("-", "_")
    
    print(f"Mapping '{card_id}' -> '{cid}'")
    
    if cid in CARD_VOCAB:
        return CARD_VOCAB[cid]
    
    # Substring search
    for k in CARD_VOCAB.keys():
        if cid in k:
            return CARD_VOCAB[k]
            
    return -1

def test_mappings():
    test_cases = [
        "One-Two Punch",
        "ONE_TWO_PUNCH",
        "STRIKE_IRONCLAD",
        "Strike Ironclad",
        {"id": "One-Two Punch", "name": "One-Two Punch"},
    ]
    
    for tc in test_cases:
        idx = get_card_idx_mock(tc)
        print(f"Result for '{tc}': {idx}")
        if idx == -1:
            print(f"FAILED: '{tc}' not found in vocab")
        else:
            print(f"SUCCESS: '{tc}' -> {idx}")
        print("-" * 20)

if __name__ == "__main__":
    test_mappings()
