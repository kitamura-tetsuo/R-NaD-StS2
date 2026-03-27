use crate::creature::{Creature, PowerIdHelper};
use crate::card::Card;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub player: Creature,
    pub enemies: Vec<Creature>,
    pub hand: Vec<Card>,
    pub draw_pile: Vec<Card>,
    pub discard_pile: Vec<Card>,
    pub exhaust_pile: Vec<Card>,
    pub energy: i32,
    pub floor: i32,
}

impl GameState {
    pub fn new(player: Creature, enemies: Vec<Creature>, energy: i32) -> Self {
        Self {
            player,
            enemies,
            hand: Vec::new(),
            draw_pile: Vec::new(),
            discard_pile: Vec::new(),
            exhaust_pile: Vec::new(),
            energy,
            floor: 1,
        }
    }

    pub fn calculate_damage(&self, attacker: &Creature, target: &Creature, base_damage: i32) -> i32 {
        let mut damage = base_damage as f32;

        // 1. Additive Modifiers (Strength)
        let strength = attacker.get_power_amount(&PowerIdHelper::strength());
        damage += strength as f32;

        // 2. Multiplicative Modifiers (Vulnerable, Weak)
        if target.get_power_amount(&PowerIdHelper::vulnerable()) > 0 {
            damage *= 1.5;
        }
        if attacker.get_power_amount(&PowerIdHelper::weak()) > 0 {
            damage *= 0.75;
        }

        // SHRINK_POWER (amount < 0) reduces damage by 33% per stack
        let shrink = attacker.get_power_amount(&PowerIdHelper::shrink());
        if shrink < 0 {
            // formula: damage * (1.0 - 0.33 * |shrink|)
            // for shrink = -1: damage * 0.67
            // for shrink = -2: damage * 0.34
            // for shrink = -3: damage * 0.01
            let multiplier = (1.0 + (0.33 * shrink as f32)).max(0.0);
            damage *= multiplier;
        }

        // Apply floor to damage
        damage.floor() as i32
    }

    pub fn play_card(&mut self, card_idx: usize, target_idx: Option<usize>) {
        if card_idx >= self.hand.len() {
            return;
        }

        let card = self.hand.remove(card_idx);
        if self.energy < card.cost {
            self.hand.insert(card_idx, card);
            return;
        }

        self.energy -= card.cost;

        // Execute card effect
        if card.is_strike() {
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                    }
                }
            }
        } else if card.is_defend() {
            let block_amount = card.base_block + self.player.get_power_amount(&PowerIdHelper::dexterity());
            self.player.add_block(block_amount);
        } else if card.is_bash() {
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                        mutable_target.add_power(PowerIdHelper::vulnerable(), card.magic_number);
                    }
                }
            }
        }

        self.discard_pile.push(card);
    }
}
