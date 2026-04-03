use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

pub mod creature;
pub mod card;
pub mod state;
pub mod vocabulary;
pub mod encoder;
pub mod search;

use crate::state::GameState;
use crate::creature::Creature;
use crate::card::{Card, TargetType};
use crate::vocabulary::Vocabulary;
use crate::search::StateEnumerator;
use std::fs::OpenOptions;
use memmap2::MmapMut;

#[pyclass]
pub struct Simulator {
    pub state: GameState,
    pub vocab: Vocabulary,
    pub shm: Option<MmapMut>,
}

#[pymethods]
impl Simulator {
    #[new]
    pub fn new(player_hp: i32, player_max_hp: i32, energy: i32, stars: i32) -> Self {
        let player = Creature::new("Player".to_string(), player_max_hp);
        let mut sim = Self {
            state: GameState::new(player, Vec::new(), energy, stars),
            vocab: Vocabulary::default(),
            shm: None,
        };
        sim.state.player.cur_hp = player_hp;
        sim
    }

    pub fn set_vocabulary(&mut self, cards: HashMap<String, i32>, monsters: HashMap<String, i32>, powers: HashMap<String, i32>, bosses: HashMap<String, i32>, potions: HashMap<String, i32>) {
        self.vocab.cards = cards;
        self.vocab.monsters = monsters;
        self.vocab.powers = powers;
        self.vocab.bosses = bosses;
        self.vocab.potions = potions;
    }

    pub fn init_shm(&mut self, path: String, size: usize) -> PyResult<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        file.set_len(size as u64).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))? };
        self.shm = Some(mmap);
        Ok(())
    }

    pub fn enumerate_final_states(&mut self) -> PyResult<usize> {
        let enumerator = StateEnumerator::new(self.state.clone(), &self.vocab);
        let results = enumerator.enumerate();
        
        if let Some(shm) = &mut self.shm {
            let mut offset = 0;
            // num_results
            shm[offset..offset+4].copy_from_slice(&(results.len() as i32).to_le_bytes());
            offset += 4;
            
            let num_results = results.len();
            for res in results {
                // num_actions
                shm[offset..offset+4].copy_from_slice(&(res.actions.len() as i32).to_le_bytes());
                offset += 4;
                // actions
                for action in res.actions {
                    shm[offset..offset+4].copy_from_slice(&action.to_le_bytes());
                    offset += 4;
                }
                // num_outcomes
                shm[offset..offset+4].copy_from_slice(&(res.outcomes.len() as i32).to_le_bytes());
                offset += 4;
                for (prob, tensor) in res.outcomes {
                    shm[offset..offset+4].copy_from_slice(&prob.to_le_bytes());
                    offset += 4;
                    for val in tensor {
                        shm[offset..offset+4].copy_from_slice(&val.to_le_bytes());
                        offset += 4;
                    }
                }
            }
            Ok(num_results)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Shared memory not initialized"))
        }
    }
    pub fn add_enemy(&mut self, id: String, hp: i32, max_hp: i32, block: i32) {
        let mut enemy = Creature::new(id, max_hp);
        enemy.cur_hp = hp;
        enemy.block = block;
        self.state.enemies.push(enemy);
    }

    pub fn add_card_to_hand(&mut self, id: String, cost: i32, damage: i32, block: i32, magic: i32, is_playable: bool) {
        let target = if id.contains("STRIKE") || id == "BASH" || id == "IRON_WAVE" || id == "HEMOKINESIS" {
            TargetType::Single
        } else if id.contains("DEFEND") {
            TargetType::SelfTarget
        } else if id == "THUNDERCLAP" || id == "CLEAVE" || id == "WHIRLWIND" {
            TargetType::All
        } else {
            TargetType::None
        };
        let card = Card::new(id, cost, damage, block, magic, target, is_playable);
        self.state.hand.push(card);
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
        Ok(Self { 
            state,
            vocab: Vocabulary::default(),
            shm: None,
        })
    }
}

#[pymodule]
fn battle_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    Ok(())
}
