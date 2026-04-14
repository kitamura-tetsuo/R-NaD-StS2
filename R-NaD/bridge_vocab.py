import os
import json
from bridge_logger import log

# --- Card Vocabulary Mapping ---
CARD_VOCAB = {
    "UNKNOWN": 0,
    "STRIKE_IRONCLAD": 1,
    "STRIKE": 1, # Alias for base Strike
    "DEFEND_IRONCLAD": 2,
    "DEFEND": 2, # Alias for base Defend
    "BASH": 3,
    "ANGER": 4,
    "BODY_SLAM": 5,
    "EXPECT_A_FIGHT": 6,
    "EXPECT_AFIGHT": 6, # Aliased for safety
    "CLASH": 6,
    "CLEAVE": 7,
    "CLOTHESLINE": 8,
    "FLEX": 9,
    "HAVOC": 10,
    "IRON_WAVE": 11,
    "PERFECTED_STRIKE": 12,
    "POMMEL_STRIKE": 13,
    "SHRUG_IT_OFF": 14,
    "SWORD_BOOMERANG": 15,
    "THUNDERCLAP": 16,
    "TRUE_GRIT": 17,
    "TWIN_STRIKE": 18,
    "WARCRY": 19,
    "WILD_STRIKE": 20,
    "ARMAMENTS": 21,
    "BLOOD_FOR_BLOOD": 22,
    "BLOOD_LETTING": 23,
    "BLOODLETTING": 23, # Alias for game ID
    "BURNING_BARRIER": 24, # StS2 specific?
    "CARNAGE": 25,
    "COMBUST": 26,
    "DARK_EMBRACE": 27,
    "DISARM": 28,
    "DUAL_WIELD": 29,
    "ENTRENCH": 30,
    "EVOLVE": 31,
    "FEEL_THE_BURN": 32,
    "FIRE_BREATHING": 33,
    "FLAME_BARRIER": 34,
    "GHOSTLY_ARMOR": 35,
    "HEMOKINESIS": 36,
    "INFERNAL_BLADE": 37,
    "INFLAME": 38,
    "INTIMIDATE": 39,
    "METALLICIZE": 40,
    "POWER_THROUGH": 41,
    "PUMMEL": 42,
    "RAGE": 43,
    "RAMPAGE": 44,
    "RECKLESS_CHARGE": 45,
    "RUPTURE": 46,
    "SEARING_BLOW": 47,
    "SECOND_WIND": 48,
    "SEEING_RED": 49,
    "SENTINEL": 50,
    "SEVER_SOUL": 51,
    "SHOCKWAVE": 52,
    "SPOT_WEAKNESS": 53,
    "UPPERCUT": 54,
    "WHIRLWIND": 55,
    "BARRICADE": 56,
    "BERSERK": 57,
    "BRUTALITY": 58,
    "CORRUPTION": 59,
    "DEMON_FORM": 60,
    "DOUBLE_TAP": 61,
    "EXHUME": 62,
    "FEED": 63,
    "FIEND_FIRE": 64,
    "HEAL": 65,
    "IMMOLATE": 66,
    "IMPERVIOUS": 67,
    "JUGGERNAUT": 68,
    "LIMIT_BREAK": 69,
    "OFFERING": 70,
    "REAPER": 71,
    "SLIMED": 72,
    "DAZED": 73,
    "VOID": 74,
    "BURN": 75,
    "WOUND": 76,
    "ASCENDERS_BANE": 77,
    "BREAKTHROUGH": 78
}

RELIC_VOCAB = {
    "UNKNOWN": 0,
    "BURNING_BLOOD": 1,
    "RING_OF_THE_SNAKE": 2,
    "CRACKED_CORE": 3,
    "PURE_WATER": 4,
    "AKABEKO": 5,
    "ANCHOR": 6,
    "ANCIENT_TEA_SET": 7,
    "ART_OF_WAR": 8,
    "BAG_OF_MARBLES": 9,
    "BAG_OF_PREPARATION": 10,
    "BLOOD_VIAL": 11,
    "BRONZE_SCALES": 12,
    "CENTENNIAL_PUZZLE": 13,
    "CERAMIC_FISH": 14,
    "DREAM_CATCHER": 15,
    "HAPPY_FLOWER": 16,
    "LANTERN": 17,
    "MEAD_WHIP": 18,
    "NUNCHAKU": 19,
    "ODDLY_SMOOTH_STONE": 20,
    "ORICHALCUM": 21,
    "PEN_NIB": 22,
    "POTION_BELT": 23,
    "PRESERVED_INSECT": 24,
    "REGAL_PILLOW": 25,
    "SMILING_MASK": 26,
    "STRAW_DOLL": 27,
    "TOY_ORNITHOPTER": 28,
    "VAJRA": 29,
    "WAR_PAINT": 30,
    "WHETSTONE": 31,
    "HEFTY_TABLET": 32
}

POWER_VOCAB = {
    "UNKNOWN": 0,
    "STRENGTH": 1,
    "DEXTERITY": 2,
    "FOCUS": 3,
    "VULNERABLE": 4,
    "WEAK": 5,
    "FRAIL": 6,
    "NO_BLOCK_NEXT_TURN": 7,
    "ARTIFACT": 8,
    "THORNS": 9,
    "METALLICIZE": 10,
    "PLATED_ARMOR": 11,
    "REGEN": 12,
    "RITUAL": 13,
    "COMBUST": 14,
    "DARK_EMBRACE": 15,
    "EVOLVE": 16,
    "FEEL_THE_BURN": 17,
    "FIRE_BREATHING": 18,
    "FLAME_BARRIER": 19,
    "MINION": 20
}

BOSS_VOCAB = {
    "UNKNOWN": 0,
    "THE_GHOST": 12
}

MONSTER_VOCAB = {
    "UNKNOWN": 0,
    "SLIME_M": 1, "SLIME_L": 2, "SLIME_S": 3,
    "GREMLIN_FAT": 4, "GREMLIN_TSAR": 5, "GREMLIN_WIZ": 6, "GREMLIN_SHIELD": 7, "GREMLIN_SNEAK": 8,
    "CULTIST": 9, "JAW_WORM": 10, "LOUSE_RED": 11, "LOUSE_GREEN": 12,
    "SLAVER_BLUE": 13, "SLAVER_RED": 14, "SCRATCHER": 15,
    "SENTRY": 16, "NOB": 17, "LAGAVULIN": 18,
    "BOOK_OF_STABBING": 19, "SLAYER_STATUE": 20, "NEMESIS": 21,
    "AUTOMATON_MINION": 22, "TORCH_HEAD": 23, "ORB_CORE": 24
}

POTION_VOCAB = {
    "UNKNOWN": 0,
    "empty": 0,
    "Fire Potion": 1,
    "Explosive Potion": 2,
    "FearPotion": 3,
    "Strength Potion": 4,
    "Dexterity Potion": 5,
    "Block Potion": 6,
    "Speed Potion": 7,
    "LiquidBronze": 8,
    "BottledCloud": 9,
    "Regen Potion": 10,
    "Swift Potion": 11,
    "Poison Potion": 12,
    "Weak Potion": 13,
    "ColorlessPotion": 14,
    "CultistPotion": 15,
    "FruitJuice": 16,
    "BloodPotion": 17,
    "ElixirPotion": 18,
    "HeartOfIron": 19,
    "GhostInAJar": 20,
    "Ambrosia": 21,
    "BlessingOfTheForge": 22,
    "DuplicationPotion": 23,
    "EssenceOfSteel": 24,
    "LiquidMemories": 25,
    "PotionOfCapacity": 26,
}

# --- Vocabulary Expansion Logic ---
def expand_vocab(base_vocab, new_ids):
    current_ids = set(base_vocab.keys())
    # Find the next available index
    next_idx = int(max(base_vocab.values()) + 1 if base_vocab else 0)
    for id_str in sorted(new_ids):
        if id_str not in current_ids:
            base_vocab[id_str] = next_idx
            next_idx += 1
    return base_vocab, next_idx

def load_game_ids():
    global CARD_VOCAB, RELIC_VOCAB, POWER_VOCAB, MONSTER_VOCAB, BOSS_VOCAB
    GAME_IDS_PATH = "/home/ubuntu/src/R-NaD-StS2/R-NaD/scripts/game_ids.json"
    if os.path.exists(GAME_IDS_PATH):
        try:
            with open(GAME_IDS_PATH, 'r') as f:
                game_ids_data = json.load(f)
            
            CARD_VOCAB, _ = expand_vocab(CARD_VOCAB, game_ids_data.get("CARDS", []))
            RELIC_VOCAB, _ = expand_vocab(RELIC_VOCAB, game_ids_data.get("RELICS", []))
            POWER_VOCAB, _ = expand_vocab(POWER_VOCAB, game_ids_data.get("POWERS", []))
            MONSTER_VOCAB, _ = expand_vocab(MONSTER_VOCAB, game_ids_data.get("MONSTERS", []))
            # Bosses in StS2 are defined in Encounters
            BOSS_VOCAB, _ = expand_vocab(BOSS_VOCAB, game_ids_data.get("ENCOUNTERS", []))
            BOSS_VOCAB["Unknown"] = 0 # Safety fallback
            
            log(f"Expanded vocabularies from {GAME_IDS_PATH}")
            log(f"CARD_VOCAB: {len(CARD_VOCAB)}, MONSTER_VOCAB: {len(MONSTER_VOCAB)}")
        except Exception as e:
            log(f"Error expanding vocabularies: {e}")

load_game_ids()

# Re-calculate or fix vocab sizes for the model
VOCAB_SIZE = max(max(CARD_VOCAB.values()) + 1, 600)
RELIC_VOCAB_SIZE = max(max(RELIC_VOCAB.values()) + 1, 300)
POWER_VOCAB_SIZE = max(max(POWER_VOCAB.values()) + 1, 280)
BOSS_VOCAB_SIZE = max(max(BOSS_VOCAB.values()) + 1, 100)
MONSTER_VOCAB_SIZE = max(max(MONSTER_VOCAB.values()) + 1, 128)

def get_reverse_vocab(vocab):
    """Inverts a vocabulary dictionary."""
    return {v: k for k, v in vocab.items()}

# Shared mappings for validation
TT_REV = {1: "AnyEnemy", 2: "AllEnemies", 3: "RandomEnemy", 0: "None", 4: "Self"}
IT_REV = {
    1: "Attack", 2: "Defense", 3: "AttackDefense", 4: "Buff", 
    5: "Debuff", 6: "StrongDebuff", 7: "Stun", 8: "StatusCard", 
    9: "Summon", 10: "CardDebuff", 11: "Heal", 12: "Escape", 
    13: "Hidden", 14: "Sleep", 15: "DeathBlow", 0: "Unknown"
}

# Node Type Mapping (NT)
NT_REV = {1: "Monster", 2: "Elite", 3: "Unknown", 4: "RestSite", 5: "Shop", 6: "Treasure", 7: "Boss", 0: "None"}
NT_MAP = {"Monster": 1, "Elite": 2, "Unknown": 3, "Event": 3, "RestSite": 4, "Rest": 4, "Shop": 5, "Treasure": 6, "Boss": 7, "None": 0}

# Reward Type Mapping (RT)
RT_REV = {1: "Gold", 2: "Card", 3: "Relic", 4: "Potion", 5: "Curse", 0: "Unknown"}
RT_MAP = {"GoldReward": 1, "Gold": 1, "Card": 2, "CardReward": 2, "Relic": 3, "RelicReward": 3, "PotionReward": 4, "Potion": 4, "Curse": 5, "Unknown": 0}

# Intent Type Mapping (IT)
IT_MAP = {
    "Attack": 1, "Defense": 2, "Defend": 2, "AttackDefense": 3, "Buff": 4, "Debuff": 5, 
    "StrongDebuff": 6, "DebuffStrong": 6, "Stun": 7, "StatusCard": 8, "Summon": 9,
    "CardDebuff": 10, "Heal": 11, "Escape": 12, "Hidden": 13, "Sleep": 14, "DeathBlow": 15, "Unknown": 0
}
