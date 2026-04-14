import os
import json
import numpy as np
from bridge_logger import log
from bridge_vocab import (
    CARD_VOCAB, RELIC_VOCAB, POWER_VOCAB, MONSTER_VOCAB, BOSS_VOCAB,
    VOCAB_SIZE, RELIC_VOCAB_SIZE, POWER_VOCAB_SIZE, BOSS_VOCAB_SIZE, MONSTER_VOCAB_SIZE,
    TT_REV, IT_REV, NT_REV, NT_MAP, RT_REV, RT_MAP, IT_MAP, get_reverse_vocab
)
from event_dict import get_event_features
import jax.numpy as jnp

# Global feature constants
VALID_TRAJECTORY_STATES = {
    "combat", "map", "rewards", "event", "rest_site", "shop", 
    "treasure", "treasure_relics", "card_reward"
}

def get_monster_idx(monster_id):
    if not monster_id:
        assert False, "monster_id is missing or empty"
    if isinstance(monster_id, dict):
        monster_id = monster_id.get("id") or monster_id.get("name")
    if not monster_id:
        assert False, "monster_id is missing or empty after dict extraction"
    mid = str(monster_id).upper().replace(" ", "_").replace("-", "_")
    for char in "!?.(),'":
        mid = mid.replace(char, "")
    
    if mid in MONSTER_VOCAB:
        return MONSTER_VOCAB[mid]
        
    for k in MONSTER_VOCAB.keys():
        if mid in k or k in mid:
            return MONSTER_VOCAB[k]
            
    assert mid in MONSTER_VOCAB, f"Unknown monster_id: {monster_id} (mapped to {mid})"
    return MONSTER_VOCAB[mid]

def get_boss_idx(boss_id):
    if not boss_id:
        assert False, "boss_id is missing or empty"
    if isinstance(boss_id, dict):
        boss_id = boss_id.get("id") or boss_id.get("name")
    if not boss_id:
        assert False, "boss_id is missing or empty after dict extraction"
    bid = str(boss_id).upper().replace(" ", "_").replace("-", "_")
    for char in "!?.(),'":
        bid = bid.replace(char, "")
    assert bid in BOSS_VOCAB, f"Unknown boss_id: {boss_id} (mapped to {bid})"
    return BOSS_VOCAB[bid]

def get_card_idx(card_id):
    if not card_id:
        assert False, "card_id is missing or empty"
    if isinstance(card_id, dict):
        card_id = card_id.get("id") or card_id.get("name")
    if not card_id:
        assert False, "card_id is missing or empty after dict extraction"
    cid = str(card_id).split('+')[0].strip().upper().replace(" ", "_").replace("-", "_")
    for char in "!?.(),'":
        cid = cid.replace(char, "")
    
    if cid in CARD_VOCAB:
        return CARD_VOCAB[cid]
        
    if cid == "STRIKE":
        for k in ["STRIKE_IRONCLAD", "STRIKE_SILENT", "STRIKE_DEFECT", "STRIKE_REGENT", "STRIKE_NECROBINDER"]:
            if k in CARD_VOCAB: return CARD_VOCAB[k]
    elif cid == "DEFEND":
        for k in ["DEFEND_IRONCLAD", "DEFEND_SILENT", "DEFEND_DEFECT", "DEFEND_REGENT", "DEFEND_NECROBINDER"]:
            if k in CARD_VOCAB: return CARD_VOCAB[k]
    
    for k in CARD_VOCAB.keys():
        if cid in k: return CARD_VOCAB[k]

    assert cid in CARD_VOCAB, f"Unknown card_id: {card_id} (mapped to {cid})"
    return CARD_VOCAB[cid]

def get_relic_idx(relic_id):
    if not relic_id: return 0
    if isinstance(relic_id, dict):
        relic_id = relic_id.get("id") or relic_id.get("name")
    if not relic_id: return 0
    rid = str(relic_id).upper().replace(" ", "_").replace("-", "_")
    for char in "!?.(),'":
        rid = rid.replace(char, "")
    if rid not in RELIC_VOCAB: return 0
    return RELIC_VOCAB[rid]

def get_power_idx(power_id):
    if not power_id: return 0
    if isinstance(power_id, dict):
        power_id = power_id.get("id") or power_id.get("name")
    if not power_id: return 0
    pid = str(power_id).upper().replace(" ", "_").replace("-", "_")
    for char in "!?.(),'":
        pid = pid.replace(char, "")
    if pid not in POWER_VOCAB: return 0
    return POWER_VOCAB[pid]

def get_list_robust(obj, key, default=None):
    if default is None: default = []
    if not obj: return default
    val = obj.get(key, default)
    if isinstance(val, str):
        val_stripped = val.strip()
        if not val_stripped: return []
        if val_stripped.startswith("[") and val_stripped.endswith("]"):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list): return parsed
            except: pass
        return default
    return val if isinstance(val, list) else default

def encode_bow(card_list):
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if isinstance(card_list, str):
        val_stripped = card_list.strip()
        if not val_stripped: return vec
        if val_stripped.startswith("["):
            try: card_list = json.loads(card_list)
            except: return vec
        else: card_list = [card_list]
            
    if not isinstance(card_list, (list, tuple)): return vec
    for cid in card_list:
        idx = get_card_idx(cid)
        if idx < VOCAB_SIZE:
            vec[idx] += 1.0
    return vec

def needs_target(card):
    target_type = card.get("targetType", "None")
    return "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type

def decode_state(encoded_dict):
    rev_card = get_reverse_vocab(CARD_VOCAB)
    rev_relic = get_reverse_vocab(RELIC_VOCAB)
    rev_monster = get_reverse_vocab(MONSTER_VOCAB)
    rev_power = get_reverse_vocab(POWER_VOCAB)
    rev_boss = get_reverse_vocab(BOSS_VOCAB)
    
    st_idx = int(encoded_dict["state_type"])
    g = encoded_dict["global"]
    decoded = {
        "floor": round(float(g[0] * 50.0)),
        "gold": round(float(g[1] * 500.0)),
        "hp": round(float(g[2] * 100.0)),
        "maxHp": round(float(g[3] * 100.0)),
        "block": round(float(g[4] * 50.0)),
        "energy": round(float(g[5] * 5.0)),
        "stars": round(float(g[6] * 10.0)),
        "potions": [bool(g[10+i]) for i in range(5)],
        "boss": rev_boss.get(round(float(g[20] * BOSS_VOCAB_SIZE)), "UNKNOWN"),
        "relics": [rev_relic.get(int(rid), f"UNKNOWN_RELIC_{int(rid)}") for rid in encoded_dict["relic_ids"] if rid > 0]
    }
    
    for pile_key in ["draw_bow", "discard_bow", "exhaust_bow", "master_bow"]:
        bow_vec = encoded_dict[pile_key]
        indices = np.where(bow_vec > 0)[0]
        pile_cards = []
        for idx in indices:
            count = int(bow_vec[idx])
            card_id = rev_card.get(idx, f"UNKNOWN_CARD_{idx}")
            pile_cards.extend([card_id] * count)
        decoded[pile_key.replace("_bow", "")] = sorted(pile_cards)

    if st_idx == 0:
        c = encoded_dict["combat"]
        decoded["draw_count"] = round(float(c[0] * 30.0))
        decoded["discard_count"] = round(float(c[1] * 30.0))
        decoded["exhaust_count"] = round(float(c[2] * 30.0))
        hand = []
        for i in range(10):
            base = 10 + i * 10
            card_idx = int(c[base])
            if card_idx > 0:
                hand.append({
                    "id": rev_card.get(card_idx, f"UNKNOWN_CARD_{card_idx}"),
                    "isPlayable": bool(c[base+1]),
                    "targetType": TT_REV.get(int(round(float(c[base+2] * 10.0))), "Unknown"),
                    "cost": round(float(c[base+3] * 5.0)),
                    "baseDamage": round(float(c[base+4] * 20.0)),
                    "baseBlock": round(float(c[base+5] * 20.0)),
                    "magicNumber": round(float(c[base+6] * 10.0)),
                    "upgraded": bool(c[base+7]),
                    "currentDamage": round(float(c[base+8] * 50.0)),
                    "currentBlock": round(float(c[base+9] * 50.0))
                })
        decoded["hand"] = hand
        enemies = []
        for i in range(5):
            base = 110 + i * 16
            if c[base] > 0:
                e = {
                    "id": rev_monster.get(int(c[base+1]), f"UNKNOWN_MONSTER_{int(c[base+1])}"),
                    "isMinion": bool(c[base+2]),
                    "hp": round(float(c[base+3] * 200.0)),
                    "maxHp": round(float(c[base+4] * 200.0)),
                    "block": round(float(c[base+5] * 50.0)),
                    "intents": []
                }
                for j in range(2):
                    it_base = base + 6 + j * 4
                    if c[it_base] > 0 or c[it_base+1] > 0:
                        e["intents"].append({
                            "type": IT_REV.get(int(round(float(c[it_base] * 10.0))), "Unknown"),
                            "damage": round(float(c[it_base+1] * 50.0)),
                            "repeats": round(float(c[it_base+2] * 5.0)),
                            "count": round(float(c[it_base+3] * 10.0))
                        })
                enemies.append(e)
        decoded["enemies"] = enemies
        p_powers = []
        for i in range(10):
            base = 200 + i * 2
            p_idx = int(c[base])
            if p_idx > 0:
                p_powers.append({
                    "id": rev_power.get(p_idx, f"UNKNOWN_POWER_{p_idx}"),
                    "amount": round(float(c[base+1] * 10.0))
                })
        decoded["player_powers"] = p_powers
        for i in range(len(enemies)):
            e_powers = []
            ebase = 220 + i * 20
            for j in range(10):
                base = ebase + j * 2
                p_idx = int(c[base])
                if p_idx > 0:
                    e_powers.append({
                        "id": rev_power.get(p_idx, f"UNKNOWN_POWER_{p_idx}"),
                        "amount": round(float(c[base+1] * 10.0))
                    })
            enemies[i]["powers"] = e_powers
        decoded["predicted_total_damage"] = float(c[320] * 50.0)
        decoded["predicted_end_block"] = float(c[321] * 50.0)
        decoded["surplus_block"] = bool(c[322])

    if st_idx == 1:
        m = encoded_dict["map"]
        map_nodes = []
        for i in range(256):
            base = i * 8
            if m[base] > 0:
                map_nodes.append({
                    "row": round(float(m[base+1] * 20.0)),
                    "col": round(float(m[base+2] * 7.0)),
                    "type": NT_REV.get(int(round(float(m[base+3] * 10.0))), "Unknown"), 
                    "is_current": bool(m[base+4])
                })
        decoded["map_nodes"] = map_nodes
    return decoded

def normalize_original_state(state):
    state_type = state.get("type", "unknown")
    player = state.get("player", {})
    rev_card = get_reverse_vocab(CARD_VOCAB)
    rev_relic = get_reverse_vocab(RELIC_VOCAB)
    rev_monster = get_reverse_vocab(MONSTER_VOCAB)
    rev_power = get_reverse_vocab(POWER_VOCAB)
    rev_boss = get_reverse_vocab(BOSS_VOCAB)

    norm = {
        "floor": int(state.get("floor", 0)),
        "gold": int(state.get("gold", 0)),
        "hp": int(player.get("hp", state.get("hp", 0))),
        "maxHp": int(player.get("maxHp", 100)),
        "block": int(player.get("block", 0)),
        "energy": int(player.get("energy", 0)),
        "stars": int(player.get("stars", 0)),
        "potions": [p.get("id") != "empty" for p in state.get("potions", [])[:5]] + [False] * (5 - len(state.get("potions", [])[:5])),
        "boss": rev_boss.get(get_boss_idx(state.get("boss") or "UNKNOWN"), "UNKNOWN"),
        "relics": [rev_relic.get(get_relic_idx(r), f"UNKNOWN_{r}") for r in (state.get("relics", []) or player.get("relics", []))[:30]]
    }

    for p_key in ["drawPile", "discardPile", "exhaustPile", "masterDeck"]:
        pile = get_list_robust(state if p_key == "masterDeck" else player, p_key)
        norm[p_key.replace("Pile", "").replace("Deck", "")] = sorted([rev_card.get(get_card_idx(c), "UNKNOWN") for c in pile])

    if state_type == "combat":
        norm["draw_count"] = len(get_list_robust(player, "drawPile"))
        norm["discard_count"] = len(get_list_robust(player, "discardPile"))
        norm["exhaust_count"] = len(get_list_robust(player, "exhaustPile"))
        hand = []
        tt_norm = {"SingleEnemy": "AnyEnemy", "AnyEnemy": "AnyEnemy", "AllEnemy": "AllEnemies", "AllEnemies": "AllEnemies", "RandomEnemy": "RandomEnemy", "None": "None", "Self": "Self"}
        for c in get_list_robust(state, "hand")[:10]:
            hand.append({
                "id": rev_card.get(get_card_idx(c.get("id")), "UNKNOWN"),
                "isPlayable": bool(c.get("isPlayable")),
                "targetType": tt_norm.get(c.get("targetType", "None"), "Unknown"),
                "cost": int(c.get("cost", 0)),
                "baseDamage": int(c.get("baseDamage", 0)),
                "baseBlock": int(c.get("baseBlock", 0)),
                "magicNumber": int(c.get("magicNumber", 0)),
                "upgraded": bool(c.get("upgraded")),
                "currentDamage": int(c.get("currentDamage", 0)),
                "currentBlock": int(c.get("currentBlock", 0))
            })
        norm["hand"] = hand
        enemies = []
        it_norm = {"Defense": "Defense", "Defend": "Defense", "AttackDefense": "AttackDefense", "Buff": "Buff", "Debuff": "Debuff", "StrongDebuff": "StrongDebuff", "DebuffStrong": "StrongDebuff", "Stun": "Stun", "StatusCard": "StatusCard", "Summon": "Summon", "CardDebuff": "CardDebuff", "Heal": "Heal", "Escape": "Heal", "Hidden": "Hidden", "Sleep": "Sleep", "DeathBlow": "DeathBlow", "Attack": "Attack", "Unknown": "Unknown"}
        for e in [en for en in get_list_robust(state, "enemies") if en.get("hp", 0) > 0][:5]:
            norm_e = {
                "id": rev_monster.get(get_monster_idx(e.get("id")), "UNKNOWN"),
                "isMinion": bool(e.get("isMinion")),
                "hp": int(e.get("hp", 0)),
                "maxHp": int(e.get("maxHp", 1)),
                "block": int(e.get("block", 0)),
                "intents": []
            }
            for it in get_list_robust(e, "intents")[:2]:
                norm_e["intents"].append({
                    "type": it_norm.get(it.get("type", "Unknown"), "Unknown"),
                    "damage": int(it.get("damage", 0)),
                    "repeats": int(it.get("repeats", 1)),
                    "count": int(it.get("count", 0))
                })
            norm_e["powers"] = [
                {"id": rev_power.get(get_power_idx(p.get("id")), "UNKNOWN"), "amount": int(p.get("amount", 0))}
                for p in get_list_robust(e, "powers")[:10]
            ]
            enemies.append(norm_e)
        norm["enemies"] = enemies
        norm["player_powers"] = [
            {"id": rev_power.get(get_power_idx(p.get("id")), "UNKNOWN"), "amount": int(p.get("amount", 0))}
            for p in get_list_robust(player, "powers")[:10]
        ]
        norm["predicted_total_damage"] = float(state.get("predicted_total_damage", 0.0))
        norm["predicted_end_block"] = float(state.get("predicted_end_block", 0.0))
        norm["surplus_block"] = bool(state.get("surplus_block"))

    if state_type == "map":
        nodes = (state.get("nodes") or [])[:256]
        nt_norm = {"Monster": "Monster", "Elite": "Elite", "Unknown": "Unknown", "Event": "Unknown", "RestSite": "RestSite", "Rest": "RestSite", "Shop": "Shop", "Treasure": "Treasure", "Boss": "Boss", "None": "None"}
        norm["map_nodes"] = [
            {
                "row": int(n.get("row", 0)),
                "col": int(n.get("col", 0)),
                "type": nt_norm.get(n.get("type", "Unknown"), "Unknown"),
                "is_current": bool(n.get("is_current"))
            } for n in nodes
        ]
    return norm

def validate_encoding(original_state, encoded_dict):
    try:
        decoded = decode_state(encoded_dict)
        expected = normalize_original_state(original_state)
        
        def recursive_compare(d_val, e_val, path="", tol=1.1):
            errs = []
            if type(d_val) != type(e_val):
                errs.append(f"{path}: Type mismatch: Decoded={type(d_val)}, Expected={type(e_val)}")
                return errs
            if isinstance(d_val, dict):
                for k, v in d_val.items():
                    if k not in e_val: errs.append(f"{path}.{k}: Missing in expected")
                    else: errs.extend(recursive_compare(v, e_val[k], f"{path}.{k}", tol))
            elif isinstance(d_val, list):
                if len(d_val) != len(e_val): errs.append(f"{path}: Length mismatch")
                else:
                    for i, (v_d, v_e) in enumerate(zip(d_val, e_val)):
                        errs.extend(recursive_compare(v_d, v_e, f"{path}[{i}]", tol))
            elif isinstance(d_val, (int, float)):
                if abs(d_val - e_val) > tol: errs.append(f"{path}: Value mismatch")
            elif d_val != e_val: errs.append(f"{path}: Value mismatch")
            return errs

        discrepancies = recursive_compare(decoded, expected)
        if discrepancies:
            log(f"WARNING: Encoding Validation Discrepancy Found!")
            for d in discrepancies: log(f"  - {d}")
            return False
        return True
    except Exception as e:
        log(f"ERROR: validate_encoding failed: {e}")
        return False

def encode_state(state):
    state_type = state.get("type", "unknown")
    type_map = {"combat": 0, "map": 1, "rewards": 2, "event": 2, "rest_site": 2, "shop": 2, "treasure": 2, "game_over": 2, "treasure_relics": 2, "card_reward": 5, "grid_selection": 3, "hand_selection": 4, "combat_waiting": 0}
    st_idx = type_map.get(state_type, 2)
    head_map = {"combat": 0, "combat_waiting": 0, "map": 1, "shop": 2, "rest_site": 3, "event": 4, "rewards": 5, "card_reward": 5, "treasure": 5, "treasure_relics": 5, "grid_selection": 5, "hand_selection": 5, "game_over": 5}
    head_idx = head_map.get(state_type, 5)
    
    global_vec = np.zeros(512, dtype=np.float32)
    global_vec[0] = state.get("floor", 0) / 50.0
    global_vec[1] = state.get("gold", 0) / 500.0
    player = state.get("player", {})
    global_vec[2] = player.get("hp", 0) / 100.0
    global_vec[3] = player.get("maxHp", 100) / 100.0
    global_vec[4] = player.get("block", 0) / 50.0
    global_vec[5] = player.get("energy", 0) / 5.0
    global_vec[6] = player.get("stars", 0) / 10.0
    potions = state.get("potions", [])
    for i in range(min(len(potions), 5)):
        if potions[i].get("id") != "empty": global_vec[10 + i] = 1.0
    boss_id = state.get("boss") or "UNKNOWN"
    global_vec[20] = get_boss_idx(boss_id) / float(BOSS_VOCAB_SIZE)
    relics = state.get("relics", []) or player.get("relics", [])
    relic_ids = np.zeros(30, dtype=np.float32)
    for i, rid in enumerate(relics[:30]):
        idx = get_relic_idx(rid)
        relic_ids[i] = idx
        if 0 < idx < 512 - 30: global_vec[30 + idx] = 1.0

    combat_vec = np.zeros(384, dtype=np.float32)
    draw_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    discard_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    exhaust_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    master_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if st_idx == 0:
        draw_pile = get_list_robust(player, "drawPile")
        discard_pile = get_list_robust(player, "discardPile")
        exhaust_pile = get_list_robust(player, "exhaustPile")
        master_deck = get_list_robust(player, "masterDeck")
        draw_bow, discard_bow, exhaust_bow, master_bow = encode_bow(draw_pile), encode_bow(discard_pile), encode_bow(exhaust_pile), encode_bow(master_deck)
        combat_vec[0], combat_vec[1], combat_vec[2] = len(draw_pile)/30.0, len(discard_pile)/30.0, len(exhaust_pile)/30.0
        hand = get_list_robust(state, "hand")
        for i in range(min(len(hand), 10)):
            card, base = hand[i], 10 + i * 10
            combat_vec[base] = get_card_idx(card.get("id", ""))
            combat_vec[base + 1] = 1.0 if card.get("isPlayable") else 0.0
            tt_map = {"AnyEnemy": 1, "SingleEnemy": 1, "AllEnemies": 2, "AllEnemy": 2, "RandomEnemy": 3, "None": 0, "Self": 4}
            combat_vec[base + 2] = tt_map.get(card.get("targetType", "None"), 0) / 10.0
            combat_vec[base + 3], combat_vec[base + 4], combat_vec[base + 5] = card.get("cost", 0)/5.0, card.get("baseDamage", 0)/20.0, card.get("baseBlock", 0)/20.0
            combat_vec[base + 6], combat_vec[base + 7] = card.get("magicNumber", 0)/10.0, (1.0 if card.get("upgraded") else 0.0)
            combat_vec[base + 8], combat_vec[base + 9] = card.get("currentDamage", 0)/50.0, card.get("currentBlock", 0)/50.0
        enemies = get_list_robust(state, "enemies")
        for i in range(min(len(enemies), 5)):
            enemy, base = enemies[i], 110 + i * 16
            combat_vec[base], combat_vec[base+1], combat_vec[base+2] = 1.0, get_monster_idx(enemy.get("id")), (1.0 if enemy.get("isMinion") else 0.0)
            combat_vec[base+3], combat_vec[base+4], combat_vec[base+5] = enemy.get("hp", 0)/200.0, enemy.get("maxHp", 1)/200.0, enemy.get("block", 0)/50.0
            for j, intent in enumerate(get_list_robust(enemy, "intents")[:2]):
                idx = base + 6 + j * 4
                combat_vec[idx] = IT_MAP.get(intent.get("type", "Unknown"), 0) / 10.0
                combat_vec[idx+1], combat_vec[idx+2], combat_vec[idx+3] = intent.get("damage", 0)/50.0, intent.get("repeats", 1)/5.0, intent.get("count", 0)/10.0
        p_powers = get_list_robust(player, "powers")
        for i in range(min(len(p_powers), 10)):
            p, base = p_powers[i], 200 + i * 2
            combat_vec[base], combat_vec[base+1] = get_power_idx(p.get("id")), p.get("amount", 0)/10.0
        for i in range(min(len(enemies), 5)):
            e_powers, base_e = get_list_robust(enemies[i], "powers"), 220 + i * 20
            for j in range(min(len(e_powers), 10)):
                p, idx = e_powers[j], base_e + j * 2
                combat_vec[idx], combat_vec[idx+1] = get_power_idx(p.get("id")), p.get("amount", 0)/10.0
        combat_vec[320], combat_vec[321], combat_vec[322] = state.get("predicted_total_damage", 0)/50.0, state.get("predicted_end_block", 0)/50.0, (1.0 if state.get("surplus_block") else 0.0)

    card_reward_vec = np.zeros(128, dtype=np.float32)
    if st_idx == 5:
        cards = state.get("cards", [])
        for i in range(min(len(cards), 5)):
            card, base = cards[i], i * 20
            card_reward_vec[base], card_id = 1.0, card.get("id") or card.get("name", "")
            card_reward_vec[base+1] = get_card_idx(card_id)
            card_reward_vec[base+2], card_reward_vec[base+3] = (1.0 if card.get("upgraded") else 0.0), card.get("cost", 0)/5.0
            card_reward_vec[base+4], card_reward_vec[base+5], card_reward_vec[base+6] = card.get("baseDamage", 0)/20.0, card.get("baseBlock", 0)/20.0, card.get("magicNumber", 0)/10.0
            card_reward_vec[base+7], card_reward_vec[base+8] = card.get("currentDamage", 0)/50.0, card.get("currentBlock", 0)/50.0
            type_m_c = {"Attack": 1, "Skill": 2, "Power": 3, "Status": 4, "Curse": 5}
            card_reward_vec[base+9] = type_m_c.get(card.get("type", ""), 0) / 5.0
            rarity_m = {"Common": 1, "Uncommon": 2, "Rare": 3, "Special": 4, "Basic": 5}
            card_reward_vec[base+10] = rarity_m.get(card.get("rarity", ""), 0) / 5.0

    map_vec = np.zeros(2048, dtype=np.float32)
    map_data = state.get("map") if isinstance(state.get("map"), dict) else state
    nodes = map_data.get("nodes", [])
    if nodes and st_idx in [1, 2, 3]:
        current_pos = map_data.get("current_pos") or {}
        for i in range(min(len(nodes), 256)):
            node, base = nodes[i], i * 8
            row, col = node.get("row", 0), node.get("col", 0)
            map_vec[base], map_vec[base+1], map_vec[base+2] = 1.0, row/20.0, col/7.0
            map_vec[base+3] = NT_MAP.get(node.get("type", "Unknown"), 0) / 10.0
            if current_pos and row == current_pos.get("row") and col == current_pos.get("col"): map_vec[base+4] = 1.0

    event_vec = np.zeros(128, dtype=np.float32)
    if st_idx == 2:
        if state_type == "rewards":
            rewards = state.get("rewards", [])
            for i in range(min(len(rewards), 10)):
                reward, base = rewards[i], i * 4
                event_vec[base], r_type = 1.0, reward.get("type", "Unknown")
                event_vec[base+1] = RT_MAP.get(r_type, 0) / 10.0
        elif state_type == "event":
            options, event_id = state.get("options", []), state.get("id", "Unknown")
            for i in range(min(len(options), 10)):
                event_vec[i] = 1.0 if not options[i].get("is_locked") else 0.5
                rich_feats = get_event_features(event_id, i)
                event_vec[20 + i * 10 : 30 + i * 10] = rich_feats
        elif state_type == "shop": event_vec[0] = 1.0
    if st_idx in [3, 4]:
        cards = state.get("cards", [])
        for i in range(min(len(cards), 10)):
            card, base = cards[i], i * 4
            event_vec[base], card_id = 1.0, card.get("id") or card.get("name", "")
            event_vec[base+1], event_vec[base+2], event_vec[base+3] = get_card_idx(card_id), (1.0 if card.get("upgraded") else 0.0), card.get("cost", 0)/5.0
    if state_type == "grid_selection": event_vec[90] = 1.0
    elif state_type == "hand_selection": event_vec[90] = -1.0

    res = {"global": global_vec, "combat": combat_vec, "relic_ids": relic_ids, "draw_bow": draw_bow, "discard_bow": discard_bow, "exhaust_bow": exhaust_bow, "master_bow": master_bow, "map": map_vec, "event": event_vec, "card_reward": card_reward_vec, "state_type": np.int32(st_idx), "head_type": np.int32(head_idx)}
    if os.environ.get("RNAD_DEBUG_ENCODING") == "1": validate_encoding(state, res)
    return res

def get_action_mask(state, route_mode=False, masked_reward_indices=None):
    mask = np.zeros(100, dtype=bool)
    state_type = state.get("type", "unknown")
    
    if state_type == "combat":
        hand = state.get("hand") or []
        enemies = state.get("enemies", []) or []
        actions_disabled = state.get("actions_disabled", False)
        
        if not actions_disabled:
            for i in range(min(len(hand), 10)):
                card = hand[i]
                if card.get("isPlayable"):
                    if needs_target(card):
                        for t in range(min(len(enemies), 5)):
                            enemy = enemies[t]
                            if enemy.get("hp", 0) > 0:
                                mask[i * 5 + t] = True
                    else:
                        mask[i * 5] = True
        
        if not actions_disabled:
            potions = state.get("potions", [])
            for i in range(min(len(potions), 5)):
                potion = potions[i]
                if potion.get("canUse", False):
                    target_type = potion.get("targetType", "None")
                    if "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type:
                        for t in range(min(len(enemies), 5)):
                            if enemies[t].get("hp", 0) > 0:
                                mask[50 + i * 5 + t] = True
                    else:
                        mask[50 + i * 5] = True
        
        if not actions_disabled and enemies:
            mask[75] = True
        
        if state.get("can_proceed"):
            mask[86] = True
    
    elif state_type == "rewards":
        rewards = state.get("rewards", [])
        has_open_potion_slots = state.get("has_open_potion_slots", True)
        potion_reward_handled = False
        for i in range(min(len(rewards), 10)):
            if masked_reward_indices and i in masked_reward_indices:
                continue
            reward = rewards[i]
            if "Potion" in reward.get("type", "") and not has_open_potion_slots:
                if not potion_reward_handled:
                    potions = state.get("potions", [])
                    for j in range(min(len(potions), 5)):
                        if potions[j].get("id") != "empty":
                            mask[94 + j] = True
                    mask[99] = True
                    potion_reward_handled = True
                continue
            mask[76 + i] = True
        if state.get("can_proceed"):
            mask[86] = True
            
    elif state_type == "map":
        next_nodes = state.get("next_nodes", [])
        if route_mode and next_nodes:
            mask[0] = True
        else:
            for i in range(min(len(next_nodes), 10)):
                mask[i] = True
        if state.get("can_proceed"):
            mask[86] = True
            
    elif state_type == "event":
        options = state.get("options", [])
        for i in range(min(len(options), 10)):
            if not options[i].get("is_locked"):
                mask[i] = True
        if state.get("can_proceed"):
            mask[86] = True
    
    elif state_type == "rest_site":
        options = state.get("options", [])
        # In Rest Site, options are indices 0-9 by default? 
        # Actually in original code it's not explicitly handled in if/else but it works.
        # Let's add it.
        for i in range(min(len(options), 10)):
            mask[i] = True
        if state.get("can_proceed"):
            mask[86] = True

    elif state_type == "shop":
        items = state.get("items", [])
        for i in range(min(len(items), 10)):
            mask[i] = True
        mask[86] = True # Proceed (leave shop)

    elif state_type == "treasure_relics":
        relics = state.get("relics", [])
        for i in range(min(len(relics), 10)):
            mask[i] = True
        if state.get("can_proceed"):
            mask[86] = True

    elif state_type == "card_reward":
        cards = state.get("cards", [])
        for i in range(min(len(cards), 5)):
            mask[i] = True
        buttons = state.get("buttons", [])
        for i in range(min(len(buttons), 5)):
            mask[10 + i] = True
        if state.get("can_proceed"):
            mask[86] = True

    elif state_type in ["grid_selection", "hand_selection"]:
        cards = state.get("cards", [])
        for i in range(min(len(cards), 20)):
            mask[i] = True
        mask[90] = True # Confirm/Skip

    elif state_type == "game_over":
        mask[86] = True # Proceed
        mask[87] = True # Return to Main Menu

    return mask

def get_semantic_map():
    return {
        "CARD_VOCAB": CARD_VOCAB,
        "RELIC_VOCAB": RELIC_VOCAB,
        "POWER_VOCAB": POWER_VOCAB,
        "BOSS_VOCAB": BOSS_VOCAB,
        "ACTION_SPACE": {
            "0-49": "Cards (up to 10 cards * 5 targets)",
            "50-74": "Potions (up to 5 potions * 5 targets)",
            "75": "End Turn",
            "76-85": "Select Reward",
            "86": "Proceed",
            "87": "Return to Main Menu",
            "88-89": "Room Selection",
            "90": "Confirm Selection / Skip (Grid)",
            "91": "Open Chest",
            "92-93": "Shop Interaction",
            "94-98": "Discard Potion",
            "99": "Wait"
        }
    }
