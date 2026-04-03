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
    #[serde(rename = "currentDamage")]
    pub current_damage: i32,
    #[serde(rename = "currentBlock")]
    pub current_block: i32,
    pub magic_number: i32,
    pub target: TargetType,
    pub is_upgraded: bool,
    #[serde(rename = "isPlayable")]
    #[serde(default = "default_true")]
    pub is_playable: bool,
}

fn default_true() -> bool {
    true
}

impl Card {
    pub fn new(id: String, cost: i32, damage: i32, block: i32, magic: i32, target: TargetType, is_playable: bool) -> Self {
        Self {
            id,
            cost,
            base_damage: damage,
            base_block: block,
            current_damage: damage,
            current_block: block,
            magic_number: magic,
            target,
            is_upgraded: false,
            is_playable,
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

    pub fn is_power(&self) -> bool {
        self.id == "FEEL_NO_PAIN" || self.id == "DARK_EMBRACE" || self.id == "CORRUPTION" || self.id == "BARRICADE" || self.id == "DEMON_FORM" || self.id == "JUGGERNAUT" || self.id == "RUPTURE" || self.id == "BRUTALITY" || self.id == "INFLAME" || self.id == "COMBUST" || self.id == "EVOLVE" || self.id == "METALLICIZE" || self.id == "FIRE_BREATHING"
    }

    pub fn is_feel_no_pain(&self) -> bool {
        self.id == "FEEL_NO_PAIN"
    }
}
