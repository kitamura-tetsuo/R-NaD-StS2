use crate::state::GameState;
use crate::vocabulary::Vocabulary;

pub const GLOBAL_SIZE: usize = 512;
pub const COMBAT_SIZE: usize = 384;
pub const BOW_SIZE: usize = 611;

pub fn encode_state_to_tensor(state: &GameState, vocab: &Vocabulary) -> Vec<f32> {
    let mut tensor = vec![0.0; GLOBAL_SIZE + COMBAT_SIZE + BOW_SIZE * 4 + 2];
    let mut offset = 0;

    // 1. Global features (512)
    tensor[offset + 0] = state.floor as f32 / 50.0;
    // tensor[offset + 1] = gold / 500.0; // Simulated gold? 
    tensor[offset + 2] = state.player.cur_hp as f32 / 100.0;
    tensor[offset + 3] = state.player.max_hp as f32 / 100.0;
    tensor[offset + 4] = state.player.block as f32 / 50.0;
    tensor[offset + 5] = state.energy as f32 / 5.0;
    tensor[offset + 6] = state.max_energy as f32 / 5.0;
    tensor[offset + 7] = state.stars as f32 / 5.0;
    tensor[offset + 8] = if state.retains_block { 1.0 } else { 0.0 };
    
    // Predicted damage/block
    tensor[offset + 10] = state.predicted_total_damage as f32 / 50.0;
    tensor[offset + 11] = state.predicted_end_block as f32 / 50.0;
    tensor[offset + 12] = (state.predicted_total_damage - state.predicted_end_block).max(0) as f32 / 50.0;

    // Boss ID (index 20)
    // tensor[offset + 20] = vocab.get_boss_idx(&state.boss) as f32 / 100.0;

    // Relics (index 30+)
    // TODO: if GameState gains relics field

    // Potions (index 50-69, 5 slots * 4 indices)
    for i in 0..state.potions.len().min(5) {
        let potion = &state.potions[i];
        let p_idx = offset + 50 + i * 4;
        tensor[p_idx] = vocab.get_potion_idx(&potion.id) as f32;
        tensor[p_idx + 1] = if potion.can_use { 1.0 } else { 0.0 };
        
        // targetType mapping (None=0, Single=0.1, All=0.2, SelfTarget=0.3)
        tensor[p_idx + 2] = match potion.target_type.as_str() {
            "Single" | "SingleEnemy" | "AnyEnemy" => 0.1,
            "All" | "AllEnemies" | "AllEnemy" => 0.2,
            "Self" | "SelfTarget" => 0.3,
            _ => 0.0,
        };
    }

    offset += GLOBAL_SIZE;

    // 2. Combat features (384)
    tensor[offset + 0] = state.draw_pile.len() as f32 / 30.0;
    tensor[offset + 1] = state.discard_pile.len() as f32 / 30.0;
    tensor[offset + 2] = state.exhaust_pile.len() as f32 / 30.0;

    // Hand cards (index 10-119, stride 11)
    for i in 0..state.hand.len().min(10) {
        let card = &state.hand[i];
        let card_idx = offset + 10 + i * 11;
        tensor[card_idx] = vocab.get_card_idx(&card.id) as f32;
        tensor[card_idx + 1] = if card.is_playable { 1.0 } else { 0.0 };
        
        // targetType mapping
        let tt_val = match card.target {
            crate::card::TargetType::Single => 1.0,
            crate::card::TargetType::All => 2.0,
            crate::card::TargetType::SelfTarget => 4.0,
            crate::card::TargetType::None => 0.0,
        };
        tensor[card_idx + 2] = tt_val / 10.0;
        tensor[card_idx + 3] = card.cost as f32 / 5.0;
        tensor[card_idx + 4] = card.base_damage as f32 / 20.0;
        tensor[card_idx + 5] = card.base_block as f32 / 20.0;
        tensor[card_idx + 6] = card.magic_number as f32 / 10.0;
        tensor[card_idx + 7] = if card.is_upgraded { 1.0 } else { 0.0 };
        tensor[card_idx + 8] = card.current_damage as f32 / 20.0;
        tensor[card_idx + 9] = card.current_block as f32 / 20.0;
        tensor[card_idx + 10] = if card.is_generated { 1.0 } else { 0.0 };
    }

    // Enemies (index 120-199)
    for i in 0..state.enemies.len().min(5) {
        let enemy = &state.enemies[i];
        let enemy_idx = offset + 120 + i * 16;
        if enemy.is_alive() {
            tensor[enemy_idx] = 1.0;
            tensor[enemy_idx + 1] = vocab.get_monster_idx(&enemy.id) as f32;
            tensor[enemy_idx + 2] = if enemy.is_minion { 1.0 } else { 0.0 };
            tensor[enemy_idx + 3] = enemy.cur_hp as f32 / 200.0;
            tensor[enemy_idx + 4] = enemy.max_hp as f32 / 200.0;
            tensor[enemy_idx + 5] = enemy.block as f32 / 50.0;
            
            // Encode intents (Indices 6-13, 8 slots)
            // Primary intent
            if let Some(primary) = enemy.intents.get(0) {
                tensor[enemy_idx + 6] = map_intent_type_to_float(&primary.intent_type);
                tensor[enemy_idx + 7] = primary.damage as f32 / 20.0;
                tensor[enemy_idx + 8] = primary.repeats as f32 / 5.0;
                tensor[enemy_idx + 9] = primary.count as f32 / 5.0;
            }
            
            // Secondary intent (if any) - common in BOSS fights or some elites
            if let Some(secondary) = enemy.intents.get(1) {
                tensor[enemy_idx + 10] = map_intent_type_to_float(&secondary.intent_type);
                tensor[enemy_idx + 11] = secondary.damage as f32 / 20.0;
                tensor[enemy_idx + 12] = secondary.repeats as f32 / 5.0;
                tensor[enemy_idx + 13] = secondary.count as f32 / 5.0;
            }
        }
    }

    // Powers (index 200+)
    for i in 0..state.player.powers.len().min(10) {
        let p = &state.player.powers[i];
        let p_idx = offset + 200 + i * 2;
        tensor[p_idx] = vocab.get_power_idx(&p.id) as f32 / 280.0; // 280 is POWER_VOCAB_SIZE
        tensor[p_idx + 1] = p.amount as f32 / 10.0;
    }

    // Enemy powers (index 220+)
    for i in 0..state.enemies.len().min(5) {
        let e = &state.enemies[i];
        let base = offset + 220 + i * 20;
        for j in 0..e.powers.len().min(10) {
            let p = &e.powers[j];
            let p_idx = base + j * 2;
            tensor[p_idx] = vocab.get_power_idx(&p.id) as f32 / 280.0;
            tensor[p_idx + 1] = p.amount as f32 / 10.0;
        }
    }

    offset += COMBAT_SIZE;

    // 3. BOW (Draw, Discard, Exhaust, Master)
    encode_bow(&state.draw_pile, &mut tensor[offset .. offset + BOW_SIZE], vocab);
    offset += BOW_SIZE;
    encode_bow(&state.discard_pile, &mut tensor[offset .. offset + BOW_SIZE], vocab);
    offset += BOW_SIZE;
    encode_bow(&state.exhaust_pile, &mut tensor[offset .. offset + BOW_SIZE], vocab);
    offset += BOW_SIZE;
    // Master deck bow? Simulator might not have it, but we should fill it if we want consistency.
    // For combat evaluation, we don't have enough master deck info here yet.
    // encode_bow(..., &mut tensor[offset .. offset + BOW_SIZE], vocab);
    offset += BOW_SIZE;

    // 4. Types
    tensor[offset] = 0.0; // state_type: 0 (Combat)
    tensor[offset + 1] = 0.0; // head_type: 0 (Combat)

    tensor
}

fn encode_bow(pile: &[crate::card::Card], buffer: &mut [f32], vocab: &Vocabulary) {
    for card in pile {
        let idx = vocab.get_card_idx(&card.id) as usize;
        if idx < BOW_SIZE {
            buffer[idx] += 1.0;
        }
    }
}

fn map_intent_type_to_float(intent_type: &str) -> f32 {
    match intent_type {
        "Attack" => 0.1,
        "Defend" => 0.2,
        "Buff" => 0.3,
        "Debuff" => 0.4,
        "StrongDebuff" => 0.5,
        "AttackDefend" => 0.6,
        "AttackBuff" => 0.7,
        "AttackDebuff" => 0.8,
        "Stun" | "Sleep" => 0.9,
        "Escape" => 1.0,
        _ => 0.0,
    }
}
