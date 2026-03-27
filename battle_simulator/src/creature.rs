use serde::{Deserialize, Serialize};


pub type PowerId = String;

pub struct PowerIdHelper;
impl PowerIdHelper {
    pub fn strength() -> PowerId { "STRENGTH_POWER".to_string() }
    pub fn dexterity() -> PowerId { "DEXTERITY_POWER".to_string() }
    pub fn vulnerable() -> PowerId { "VULNERABLE_POWER".to_string() }
    pub fn weak() -> PowerId { "WEAK_POWER".to_string() }
    pub fn frail() -> PowerId { "FRAIL_POWER".to_string() }
    pub fn shrink() -> PowerId { "SHRINK_POWER".to_string() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Power {
    pub id: PowerId,
    pub amount: i32,
}

pub type Intent = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Creature {
    pub id: String,
    pub max_hp: i32,
    pub cur_hp: i32,
    pub block: i32,
    pub powers: Vec<Power>,
    pub intent: Option<Intent>,
}

impl Creature {
    pub fn new(id: String, max_hp: i32) -> Self {
        Self {
            id,
            max_hp,
            cur_hp: max_hp,
            block: 0,
            powers: Vec::new(),
            intent: None,
        }
    }

    pub fn get_power_amount(&self, id: &PowerId) -> i32 {
        self.powers.iter()
            .find(|p| &p.id == id)
            .map(|p| p.amount)
            .unwrap_or(0)
    }

    pub fn add_power(&mut self, id: PowerId, amount: i32) {
        if let Some(p) = self.powers.iter_mut().find(|p| p.id == id) {
            p.amount += amount;
        } else {
            self.powers.push(Power { id, amount });
        }
        // Remove power if amount reaches 0 and it's not a permanent power
        self.powers.retain(|p| p.amount != 0);
    }

    pub fn apply_damage(&mut self, amount: i32) -> i32 {
        let mut remaining_damage = amount;
        if self.block > 0 {
            let blocked = self.block.min(remaining_damage);
            self.block -= blocked;
            remaining_damage -= blocked;
        }
        self.cur_hp = (self.cur_hp - remaining_damage).max(0);
        remaining_damage
    }

    pub fn add_block(&mut self, amount: i32) {
        self.block += amount;
    }

    pub fn is_alive(&self) -> bool {
        self.cur_hp > 0
    }
}
