use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Vocabulary {
    pub cards: HashMap<String, i32>,
    pub monsters: HashMap<String, i32>,
    pub powers: HashMap<String, i32>,
    pub bosses: HashMap<String, i32>,
    pub potions: HashMap<String, i32>,
}

impl Vocabulary {
    pub fn get_card_idx(&self, id: &str) -> i32 {
        self.cards.get(id).copied().unwrap_or(0)
    }
    pub fn get_monster_idx(&self, id: &str) -> i32 {
        self.monsters.get(id).copied().unwrap_or(0)
    }
    pub fn get_power_idx(&self, id: &str) -> i32 {
        self.powers.get(id).copied().unwrap_or(0)
    }
    pub fn get_boss_idx(&self, id: &str) -> i32 {
        self.bosses.get(id).copied().unwrap_or(0)
    }
    pub fn get_potion_idx(&self, id: &str) -> i32 {
        self.potions.get(id).copied().unwrap_or(0)
    }
}
