use pyo3::prelude::*;
use serde_json;

pub mod creature;
pub mod card;
pub mod state;

use crate::state::GameState;
use crate::creature::Creature;
use crate::card::{Card, TargetType};

#[pyclass]
pub struct Simulator {
    pub state: GameState,
}

#[pymethods]
impl Simulator {
    #[new]
    pub fn new(player_hp: i32, player_max_hp: i32, energy: i32) -> Self {
        let player = Creature::new("Player".to_string(), player_max_hp);
        let mut sim = Self {
            state: GameState::new(player, Vec::new(), energy),
        };
        sim.state.player.cur_hp = player_hp;
        sim
    }

    pub fn add_enemy(&mut self, id: String, hp: i32, max_hp: i32, block: i32) {
        let mut enemy = Creature::new(id, max_hp);
        enemy.cur_hp = hp;
        enemy.block = block;
        self.state.enemies.push(enemy);
    }

    pub fn add_card_to_hand(&mut self, id: String, cost: i32, damage: i32, block: i32, magic: i32) {
        let target = if id.contains("STRIKE") || id == "BASH" {
            TargetType::Single
        } else if id.contains("DEFEND") {
            TargetType::SelfTarget
        } else {
            TargetType::None
        };
        self.state.hand.push(Card::new(id, cost, damage, block, magic, target));
    }

    pub fn add_power(&mut self, creature_idx: i32, power_id: String, amount: i32) {
        if creature_idx == -1 {
            self.state.player.add_power(power_id, amount);
        } else if let Some(enemy) = self.state.enemies.get_mut(creature_idx as usize) {
            enemy.add_power(power_id, amount);
        }
    }

    pub fn play_card(&mut self, card_idx: usize, target_idx: i32) {
        let t_idx = if target_idx == -1 { None } else { Some(target_idx as usize) };
        self.state.play_card(card_idx, t_idx);
    }

    pub fn set_energy(&mut self, energy: i32) {
        self.state.energy = energy;
    }

    pub fn get_state_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.state).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[staticmethod]
    pub fn from_json(json_str: String) -> PyResult<Self> {
        let state: GameState = serde_json::from_str(&json_str).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { state })
    }
}

#[pymodule]
fn battle_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    Ok(())
}
