use crate::state::GameState;
use crate::vocabulary::Vocabulary;
use crate::encoder::encode_state_to_tensor;
use std::collections::{HashSet, VecDeque};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumerateResult {
    pub actions: Vec<i32>, // Sequence of action indices
    pub outcomes: Vec<(f32, Vec<f32>)>, // (Probability, Encoded Tensor)
}

pub struct StateEnumerator<'a> {
    pub initial_state: GameState,
    pub vocab: &'a Vocabulary,
}

impl<'a> StateEnumerator<'a> {
    pub fn new(state: GameState, vocab: &'a Vocabulary) -> Self {
        Self { initial_state: state, vocab }
    }

    pub fn enumerate(&self) -> Vec<EnumerateResult> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((self.initial_state.clone(), Vec::new()));

        while let Some((state, actions)) = queue.pop_front() {
            let state_hash = self.get_state_hash(&state);
            if visited.contains(&(state_hash.clone(), actions.len())) {
                continue;
            }
            visited.insert((state_hash, actions.len()));

            let possible_actions = self.get_possible_actions(&state);
            
            if possible_actions.is_empty() {
                // End turn
                let mut final_actions = actions.clone();
                final_actions.push(75);
                results.push(EnumerateResult {
                    actions: final_actions,
                    outcomes: vec![(1.0, encode_state_to_tensor(&state, self.vocab))],
                });
                continue;
            }

            for action_idx in possible_actions {
                let card_idx = (action_idx / 5) as usize;
                let target_idx = (action_idx % 5) as usize;
                let card = &state.hand[card_idx];

                if card.is_random() {
                    // Random action: this becomes a final action sequence
                    let mut final_actions = actions.clone();
                    final_actions.push(action_idx);
                    
                    let outcomes = self.get_random_outcomes(&state, card_idx, target_idx);
                    results.push(EnumerateResult {
                        actions: final_actions,
                        outcomes,
                    });
                } else {
                    let mut next_state = state.clone();
                    next_state.play_card(card_idx, Some(target_idx));
                    
                    let mut next_actions = actions.clone();
                    next_actions.push(action_idx);
                    queue.push_back((next_state, next_actions));
                }
            }
        }

        self.prune_results(results)
    }

    fn get_possible_actions(&self, state: &GameState) -> Vec<i32> {
        let mut actions = Vec::new();
        for (i, card) in state.hand.iter().enumerate() {
            if state.energy >= card.cost {
                match card.target {
                    crate::card::TargetType::Single => {
                        for (j, enemy) in state.enemies.iter().enumerate() {
                            if enemy.is_alive() {
                                actions.push((i * 5 + j) as i32);
                            }
                        }
                    }
                    _ => {
                        actions.push((i * 5) as i32); 
                    }
                }
            }
        }
        actions
    }

    fn get_random_outcomes(&self, state: &GameState, card_idx: usize, target_idx: usize) -> Vec<(f32, Vec<f32>)> {
        let card = &state.hand[card_idx];
        let mut outcomes = Vec::new();

        if card.id == "HAVOC" {
            // Draw top card and play it for free
            // If draw_pile is non-empty, each card has 1/N prob
            if !state.draw_pile.is_empty() {
                let prob = 1.0 / state.draw_pile.len() as f32;
                for _i in 0..state.draw_pile.len() {
                    let next_state = state.clone();
                    // Simplified: Havoc play logic
                    // next_state.play_card_from_draw(i); 
                    outcomes.push((prob, encode_state_to_tensor(&next_state, self.vocab)));
                }
            } else if !state.discard_pile.is_empty() {
                // Shuffle discard into draw and then draw
                // Same logic but with discard pile
                let prob = 1.0 / state.discard_pile.len() as f32;
                for _i in 0..state.discard_pile.len() {
                    let next_state = state.clone();
                    outcomes.push((prob, encode_state_to_tensor(&next_state, self.vocab)));
                }
            } else {
                // Nothing happens
                let mut next_state = state.clone();
                next_state.play_card(card_idx, Some(target_idx));
                outcomes.push((1.0, encode_state_to_tensor(&next_state, self.vocab)));
            }
        } else {
            // Default: treat as deterministic for now if unknown random card
            let mut next_state = state.clone();
            next_state.play_card(card_idx, Some(target_idx));
            outcomes.push((1.0, encode_state_to_tensor(&next_state, self.vocab)));
        }

        outcomes
    }

    fn get_state_hash(&self, state: &GameState) -> String {
        format!("{:?}_{:?}_{:?}", state.player, state.enemies, state.energy)
    }

    fn prune_results(&self, results: Vec<EnumerateResult>) -> Vec<EnumerateResult> {
        let mut unique_results = Vec::new();
        let mut seen_states = HashSet::new();

        for res in results {
            // Use outcome tensors as part of state identity for pruning
            let mut outcome_sum = 0.0;
            for (p, t) in &res.outcomes {
                outcome_sum += p * t.iter().sum::<f32>(); // Very crude hash
            }
            let key = format!("{:.4}", outcome_sum); 
            if !seen_states.contains(&key) {
                seen_states.insert(key);
                unique_results.push(res);
            }
        }
        unique_results
    }
}
