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
    pub stars: i32,
    pub floor: i32,
}

impl GameState {
    pub fn new(player: Creature, enemies: Vec<Creature>, energy: i32, stars: i32) -> Self {
        Self {
            player,
            enemies,
            hand: Vec::new(),
            draw_pile: Vec::new(),
            discard_pile: Vec::new(),
            exhaust_pile: Vec::new(),
            energy,
            stars,
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

        // Per-hit damage capping (Slippery)
        let slippery = target.get_power_amount(&PowerIdHelper::slippery());
        if slippery > 0 {
            damage = 1.0;
        }

        // Apply floor to damage
        damage.floor() as i32
    }

    pub fn perform_aoe_damage(&mut self, base_damage: i32) {
        for t_idx in 0..self.enemies.len() {
            if let Some(target) = self.enemies.get(t_idx) {
                let dmg = self.calculate_damage(&self.player, target, base_damage);
                if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                    mutable_target.apply_damage(dmg);
                }
            }
        }
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
        if card.id == "IRON_WAVE" {
            // In StS2, IRON_WAVE is energy-neutral or costs Stars.
            // Based on observed discrepancies (real=1, sim=0 from start energy 1), it is energy-neutral.
            self.energy += card.cost;
            
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                    }
                }
            }
            // Apply Block
            let mut block_amount = (card.base_block + self.player.get_power_amount(&PowerIdHelper::dexterity())).max(0) as f32;
            if self.player.get_power_amount(&PowerIdHelper::frail()) > 0 {
                block_amount *= 0.75;
            }
            self.player.add_block(block_amount.floor() as i32);
        } else if card.id == "TWIN_STRIKE" {
            let hits = 2;
            for _ in 0..hits {
                if let Some(t_idx) = target_idx {
                    if let Some(target) = self.enemies.get(t_idx) {
                        let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                        if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                            mutable_target.apply_damage(dmg);
                        }
                    }
                }
            }
        } else if card.id == "PUMMEL" {
            let hits = if card.magic_number > 0 { card.magic_number } else if card.is_upgraded { 5 } else { 4 };
            for _ in 0..hits {
                if let Some(t_idx) = target_idx {
                    if let Some(target) = self.enemies.get(t_idx) {
                        let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                        if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                            mutable_target.apply_damage(dmg);
                        }
                    }
                }
            }
        } else if card.is_strike() || card.is_bash() {
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let mut base_dmg = card.base_damage;
                    
                    if card.id == "PERFECTED_STRIKE" {
                        let mut strike_count = 0;
                        for c in &self.hand { if c.is_strike() { strike_count += 1; } }
                        for c in &self.draw_pile { if c.is_strike() { strike_count += 1; } }
                        for c in &self.discard_pile { if c.is_strike() { strike_count += 1; } }
                        for c in &self.exhaust_pile { if c.is_strike() { strike_count += 1; } }
                        
                        // Already removed from hand but not yet in discard, so we add 1 back if it were a strike (it is)
                        strike_count += 1; 

                        let bonus = if card.is_upgraded { 3 } else { 2 };
                        base_dmg = 6 + (strike_count * bonus);
                    }

                    let dmg = self.calculate_damage(&self.player, target, base_dmg);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                        if card.is_bash() {
                            mutable_target.add_power(PowerIdHelper::vulnerable(), if card.magic_number > 0 { card.magic_number } else { 2 });
                        }
                    }
                }
            }
        } else if card.id == "SWORD_BOOMERANG" {
            let hits = if card.magic_number > 0 { 
                card.magic_number 
            } else if card.is_upgraded {
                4
            } else {
                3
            };
            for _ in 0..hits {
                // For random cards, pick a random alive enemy. 
                // For simplicity and matching validation (which often sends a target), we try the provided target first if it's alive.
                let mut t_idx_to_use = target_idx;
                
                // If no target provided or target is dead, pick first alive enemy
                if t_idx_to_use.is_none() || !self.enemies.get(t_idx_to_use.unwrap()).map_or(false, |e| e.is_alive()) {
                    t_idx_to_use = self.enemies.iter().position(|e| e.is_alive());
                }

                if let Some(t_idx) = t_idx_to_use {
                    if let Some(target) = self.enemies.get(t_idx) {
                        let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                        if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                            mutable_target.apply_damage(dmg);
                        }
                    }
                }
            }
        } else if card.id == "THUNDERCLAP" {
            let vul_amount = if card.magic_number > 0 { card.magic_number } else { 1 };
            for t_idx in 0..self.enemies.len() {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                        mutable_target.add_power(PowerIdHelper::vulnerable(), vul_amount);
                    }
                }
            }
        } else if card.id == "FIGHT_ME" {
            if let Some(t_idx) = target_idx {
                if let Some(_target) = self.enemies.get(t_idx) {
                    let hits = 2;
                    for _ in 0..hits {
                        if let Some(target) = self.enemies.get(t_idx) {
                            if target.is_alive() {
                                let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                                if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                                    mutable_target.apply_damage(dmg);
                                }
                            }
                        }
                    }
                    
                    // Player gains 2 Strength
                    self.player.add_power(PowerIdHelper::strength(), 2);
                    
                    // Enemy gains 1 Strength (if still alive)
                    if let Some(target) = self.enemies.get_mut(t_idx) {
                        if target.is_alive() {
                            target.add_power(PowerIdHelper::strength(), 1);
                        }
                    }
                }
            }
        } else if card.id == "WHIRLWIND" {
            let hits = self.energy;
            for _ in 0..hits {
                self.perform_aoe_damage(card.base_damage);
            }
            self.energy = 0;
        } else if card.id == "CLEAVE" {
            self.perform_aoe_damage(card.base_damage);
        } else if card.id == "ANGER" {
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                    }
                    // Anger effect: Add a copy to discard pile.
                    // The original card will be pushed to discard at the end of this function.
                    self.discard_pile.push(card.clone());
                }
            }
        } else if card.id == "BLOODLETTING" {
            self.player.lose_hp(3);
            self.energy += if card.magic_number > 0 { card.magic_number } else if card.is_upgraded { 3 } else { 2 };
        } else if card.id == "BODY_SLAM" {
            if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    let dmg = self.calculate_damage(&self.player, target, self.player.block);
                    if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                        mutable_target.apply_damage(dmg);
                    }
                }
            }
        } else if card.id == "BREAKTHROUGH" {
            self.player.lose_hp(1);
            self.perform_aoe_damage(card.base_damage);
        } else if card.id == "FEEL_NO_PAIN" {
            // FEEL_NO_PAIN is energy-neutral in STS2 context based on discrepancies.
            self.energy += card.cost;
            let amount = if card.magic_number > 0 { card.magic_number } else if card.is_upgraded { 4 } else { 3 };
            self.player.add_power("FEEL_NO_PAIN_POWER".to_string(), amount);
        } else if card.is_defend() {
            let mut block_amount = (card.base_block + self.player.get_power_amount(&PowerIdHelper::dexterity())).max(0) as f32;
            if self.player.get_power_amount(&PowerIdHelper::frail()) > 0 {
                block_amount *= 0.75;
            }
            self.player.add_block(block_amount.floor() as i32);
        } else {
            // Default handler for simple cards (one-hit damage or block)
            if matches!(card.target, crate::card::TargetType::All) {
                self.perform_aoe_damage(card.base_damage);
            } else if let Some(t_idx) = target_idx {
                if let Some(target) = self.enemies.get(t_idx) {
                    if card.base_damage > 0 {
                        let dmg = self.calculate_damage(&self.player, target, card.base_damage);
                        if let Some(mutable_target) = self.enemies.get_mut(t_idx) {
                            mutable_target.apply_damage(dmg);
                        }
                    }
                }
            }
            if card.base_block > 0 {
                let mut block_amount = (card.base_block + self.player.get_power_amount(&PowerIdHelper::dexterity())).max(0) as f32;
                if self.player.get_power_amount(&PowerIdHelper::frail()) > 0 {
                    block_amount *= 0.75;
                }
                self.player.add_block(block_amount.floor() as i32);
            }
        }
        if !card.is_power() {
            self.discard_pile.push(card);
        }
        self.enemies.retain(|e| e.is_alive());
    }
}
