import numpy as np
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add R-NaD to path
BRIDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

# Mock environment before importing rnad_bridge
import jax.numpy as jnp
# We need to ensure some constants are available if they are used at module level
os.environ["RNAD_RUN_ID"] = "test_run"
os.environ["SKIP_RNAD_INIT"] = "1"

import rnad_bridge
rnad_bridge.do_deferred_imports()

class TestStateEncoding(unittest.TestCase):
    def setUp(self):
        # Reset any global state if necessary
        pass

    def test_global_vector(self):
        """Test global vector encoding (indices 0-511)."""
        state = {
            "type": "combat",
            "floor": 10,
            "gold": 250,
            "player": {
                "hp": 50,
                "maxHp": 100,
                "block": 10,
                "energy": 3,
                "stars": 5,
                "relics": ["BURNING_BLOOD", "AKABEKO"]
            },
            "potions": [
                {"id": "Fire Potion"},
                {"id": "empty"},
                {"id": "Block Potion"}
            ],
            "boss": "CEREMONIAL_BEAST_BOSS"
        }
        
        encoded = rnad_bridge.encode_state(state)
        global_vec = encoded["global"]
        
        self.assertEqual(global_vec.shape, (512,))
        self.assertAlmostEqual(global_vec[0], 10/50.0)
        self.assertAlmostEqual(global_vec[1], 250/500.0)
        self.assertAlmostEqual(global_vec[2], 50/100.0)
        self.assertAlmostEqual(global_vec[3], 100/100.0)
        self.assertAlmostEqual(global_vec[4], 10/50.0)
        self.assertAlmostEqual(global_vec[5], 3/5.0)
        self.assertAlmostEqual(global_vec[6], 5/10.0)
        
        # Potions (indices 10-14)
        self.assertEqual(global_vec[10], 1.0) # Fire Potion
        self.assertEqual(global_vec[11], 0.0) # empty
        self.assertEqual(global_vec[12], 1.0) # Block Potion
        
        # Boss (index 20)
        boss_idx = rnad_bridge.get_boss_idx("CEREMONIAL_BEAST_BOSS")
        self.assertAlmostEqual(global_vec[20], boss_idx / float(rnad_bridge.BOSS_VOCAB_SIZE))
        
        # Relics (indices 30+)
        relic1_idx = rnad_bridge.get_relic_idx("BURNING_BLOOD")
        relic2_idx = rnad_bridge.get_relic_idx("AKABEKO")
        self.assertEqual(global_vec[30 + relic1_idx], 1.0)
        self.assertEqual(global_vec[30 + relic2_idx], 1.0)

    def test_combat_vector_piles_and_hand(self):
        """Test combat vector encoding for piles and hand (indices 0-109)."""
        state = {
            "type": "combat",
            "player": {
                "drawPile": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}],
                "discardPile": [{"id": "BASH"}],
                "exhaustPile": [],
                "masterDeck": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}, {"id": "BASH"}]
            },
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "isPlayable": True,
                    "targetType": "SingleEnemy",
                    "cost": 1,
                    "baseDamage": 6,
                    "baseBlock": 0,
                    "magicNumber": 0,
                    "upgraded": False,
                    "currentDamage": 6,
                    "currentBlock": 0
                }
            ],
            "enemies": [{"id": "LIVING_SHIELD", "hp": 10, "maxHp": 10, "block": 0, "intents": []}],
            "predicted_total_damage": 6,
            "predicted_end_block": 0,
            "surplus_block": False
        }
        
        encoded = rnad_bridge.encode_state(state)
        combat_vec = encoded["combat"]
        
        self.assertEqual(combat_vec.shape, (384,))
        # Pile counts (normalized)
        self.assertAlmostEqual(combat_vec[0], 2/30.0)
        self.assertAlmostEqual(combat_vec[1], 1/30.0)
        self.assertAlmostEqual(combat_vec[2], 0/30.0)
        
        # Hand (starts at index 10)
        # Card 0
        card0_base = 10
        self.assertEqual(combat_vec[card0_base], rnad_bridge.get_card_idx("STRIKE_IRONCLAD"))
        self.assertEqual(combat_vec[card0_base + 1], 1.0) # Playable
        self.assertAlmostEqual(combat_vec[card0_base + 2], 1/10.0) # SingleEnemy -> 1
        self.assertAlmostEqual(combat_vec[card0_base + 3], 1/5.0) # Cost 1
        self.assertAlmostEqual(combat_vec[card0_base + 4], 6/20.0) # Base damage 6
        self.assertAlmostEqual(combat_vec[card0_base + 7], 0.0) # Upgraded
        self.assertAlmostEqual(combat_vec[card0_base + 8], 6/50.0) # Current damage 6

        # Bows
        self.assertEqual(encoded["draw_bow"].sum(), 2.0)
        self.assertEqual(encoded["discard_bow"].sum(), 1.0)
        self.assertEqual(encoded["master_bow"].sum(), 3.0)

    def test_combat_vector_enemies_and_powers(self):
        """Test combat vector encoding for enemies and powers (indices 110-319)."""
        state = {
            "type": "combat",
            "player": {"powers": [{"id": "Strength", "amount": 3}]},
            "enemies": [
                {
                    "id": "LIVING_SHIELD",
                    "hp": 20,
                    "maxHp": 50,
                    "block": 5,
                    "isMinion": False,
                    "intents": [{"type": "Attack", "damage": 10, "repeats": 1, "count": 1}],
                    "powers": [{"id": "Ritual", "amount": 1}]
                }
            ]
        }
        
        encoded = rnad_bridge.encode_state(state)
        combat_vec = encoded["combat"]
        
        # Enemy 0 (starts at index 110)
        enemy0_base = 110
        self.assertEqual(combat_vec[enemy0_base], 1.0) # Alive
        self.assertEqual(combat_vec[enemy0_base + 1], rnad_bridge.get_monster_idx("LIVING_SHIELD"))
        self.assertEqual(combat_vec[enemy0_base + 2], 0.0) # IsMinion
        self.assertAlmostEqual(combat_vec[enemy0_base + 3], 20/200.0) # HP
        self.assertAlmostEqual(combat_vec[enemy0_base + 5], 5/50.0) # Block
        
        # Intent (starts at 110 + 6 = 116)
        intent0_base = 116
        self.assertAlmostEqual(combat_vec[intent0_base], 1/10.0) # Attack -> 1
        self.assertAlmostEqual(combat_vec[intent0_base + 1], 10/50.0) # Damage
        self.assertAlmostEqual(combat_vec[intent0_base + 2], 1/5.0) # Repeats
        
        # Player Powers (starts at 200)
        self.assertAlmostEqual(combat_vec[200], rnad_bridge.get_power_idx("Strength") / float(rnad_bridge.POWER_VOCAB_SIZE))
        self.assertAlmostEqual(combat_vec[201], 3/10.0)
        
        # Enemy 0 Powers (starts at 220)
        self.assertAlmostEqual(combat_vec[220], rnad_bridge.get_power_idx("Ritual") / float(rnad_bridge.POWER_VOCAB_SIZE))
        self.assertAlmostEqual(combat_vec[221], 1/10.0)

    def test_map_vector(self):
        """Test map vector encoding."""
        state = {
            "type": "map",
            "nodes": [
                {"row": 0, "col": 3, "type": "Monster"},
                {"row": 1, "col": 2, "type": "Elite"}
            ],
            "current_pos": {"row": 0, "col": 3}
        }
        
        encoded = rnad_bridge.encode_state(state)
        map_vec = encoded["map"]
        
        self.assertEqual(map_vec.shape, (2048,))
        # Node 0
        self.assertEqual(map_vec[0], 1.0) # Presence
        self.assertAlmostEqual(map_vec[1], 0/20.0) # Row
        self.assertAlmostEqual(map_vec[2], 3/7.0) # Col
        self.assertAlmostEqual(map_vec[3], 1/10.0) # Monster -> 1
        self.assertEqual(map_vec[4], 1.0) # IsCurrent
        
        # Node 1
        self.assertEqual(map_vec[8], 1.0)
        self.assertAlmostEqual(map_vec[9], 1/20.0)
        self.assertAlmostEqual(map_vec[11], 2/10.0) # Elite -> 2
        self.assertEqual(map_vec[12], 0.0) # Not current

    def test_event_and_reward_vector(self):
        """Test event and reward vector encoding."""
        # Rewards test
        state_rewards = {
            "type": "rewards",
            "rewards": [
                {"type": "GoldReward"},
                {"type": "Relic"},
                {"type": "CardReward"}
            ]
        }
        encoded_rewards = rnad_bridge.encode_state(state_rewards)
        event_vec = encoded_rewards["event"]
        
        # Reward 0
        self.assertEqual(event_vec[0], 1.0) # Presence
        self.assertAlmostEqual(event_vec[1], 1/10.0) # GoldReward -> 1
        # Reward 1
        self.assertEqual(event_vec[4], 1.0)
        self.assertAlmostEqual(event_vec[5], 3/10.0) # Relic -> 3
        
        # Grid selection test
        state_grid = {
            "type": "grid_selection",
            "cards": [
                {"id": "BASH", "upgraded": True, "cost": 2}
            ]
        }
        encoded_grid = rnad_bridge.encode_state(state_grid)
        grid_vec = encoded_grid["event"]
        self.assertEqual(grid_vec[0], 1.0) # Presence
        self.assertEqual(grid_vec[1], rnad_bridge.get_card_idx("BASH"))
        self.assertEqual(grid_vec[2], 1.0) # Upgraded
        self.assertAlmostEqual(grid_vec[3], 2/5.0) # Cost 2
        self.assertEqual(grid_vec[90], 1.0) # Differentiation flag for grid

    def test_action_mask(self):
        """Test action mask generation."""
        # Combat mask
        state_combat = {
            "type": "combat",
            "hand": [
                {"isPlayable": True, "targetType": "SingleEnemy"},
                {"isPlayable": False, "targetType": "Self"}
            ],
            "enemies": [{"hp": 10}, {"hp": 0}], # Enemy 0 alive, Enemy 1 dead
            "potions": [{"canUse": True, "targetType": "None"}],
            "can_proceed": False
        }
        mask = rnad_bridge.get_action_mask(state_combat)
        self.assertEqual(mask.shape, (100,))
        
        # Card 0 is playable on Enemy 0 (target index 0)
        self.assertTrue(mask[0]) # Card 0, Target 0
        self.assertFalse(mask[1]) # Card 0, Target 1 (Enemy 1 is dead)
        
        # Card 1 is not playable
        self.assertFalse(mask[5]) # Card 1, Target 0
        
        # Potion 0 is usable (starts at index 50)
        self.assertTrue(mask[50]) # Potion 0
        
        # End turn (index 75)
        self.assertTrue(mask[75])
        
        # Map mask
        state_map = {
            "type": "map",
            "next_nodes": [{}, {}]
        }
        mask_map = rnad_bridge.get_action_mask(state_map)
        self.assertTrue(mask_map[0])
        self.assertTrue(mask_map[1])
        self.assertFalse(mask_map[2])

    def test_selection_masking(self):
        """Test that already selected cards are masked out."""
        # Grid selection
        state_grid = {
            "type": "grid_selection",
            "cards": [
                {"id": "BASH", "selected": True},
                {"id": "STRIKE", "selected": False}
            ]
        }
        mask_grid = rnad_bridge.get_action_mask(state_grid)
        self.assertFalse(mask_grid[0]) # Card 0 is selected
        self.assertTrue(mask_grid[1])  # Card 1 is not selected
        
        # Hand selection
        state_hand = {
            "type": "hand_selection",
            "cards": [
                {"name": "Bash", "selected": False},
                {"name": "Strike", "selected": True}
            ]
        }
        mask_hand = rnad_bridge.get_action_mask(state_hand)
        self.assertTrue(mask_hand[0])  # Card 0 is not selected
        self.assertFalse(mask_hand[1]) # Card 1 is selected

if __name__ == "__main__":
    unittest.main()
