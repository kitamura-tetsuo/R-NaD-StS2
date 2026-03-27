use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetType {
    Single,
    All,
    SelfTarget,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Card {
    pub id: String,
    pub cost: i32,
    pub base_damage: i32,
    pub base_block: i32,
    pub magic_number: i32,
    pub target: TargetType,
    pub is_upgraded: bool,
}

impl Card {
    pub fn new(id: String, cost: i32, damage: i32, block: i32, magic: i32, target: TargetType) -> Self {
        Self {
            id,
            cost,
            base_damage: damage,
            base_block: block,
            magic_number: magic,
            target,
            is_upgraded: false,
        }
    }

    pub fn is_strike(&self) -> bool {
        self.id.contains("STRIKE")
    }

    pub fn is_defend(&self) -> bool {
        self.id.contains("DEFEND")
    }

    pub fn is_bash(&self) -> bool {
        self.id == "BASH"
    }

    pub fn is_random(&self) -> bool {
        self.id == "HAVOC" || self.id == "INFERNAL_BLADE" || self.id == "SWORD_BOOMERANG"
    }
    
    pub fn is_choice(&self) -> bool {
        false 
    }
}
