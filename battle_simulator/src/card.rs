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

    // Factory methods for common cards (Ironclad example)
    pub fn strike() -> Self {
        Self::new("STRIKE_IRONCLAD".to_string(), 1, 6, 0, 0, TargetType::Single)
    }

    pub fn defend() -> Self {
        Self::new("DEFEND_IRONCLAD".to_string(), 1, 0, 5, 0, TargetType::SelfTarget)
    }

    pub fn bash() -> Self {
        Self::new("BASH".to_string(), 2, 8, 0, 2, TargetType::Single)
    }
}
