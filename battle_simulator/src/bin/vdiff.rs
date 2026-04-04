use serde::Deserialize;
use battle_simulator::state::GameState;
use battle_simulator::creature::{Creature, Power};
use battle_simulator::card::{Card, TargetType};
use std::fs;
use clap::Parser;
use colored::*;
use similar::{ChangeTag, TextDiff};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(help = "Path to state_before.json")]
    state_before: String,
    #[arg(help = "Path to action.json")]
    action: String,
    #[arg(help = "Path to state_after.json")]
    state_after: String,
    #[arg(short, long, help = "Show full diff even if basic stats match")]
    full: bool,
}

// C# formats (as captured in discrepancy logs)
#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsPower { id: String, amount: i32 }

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsPlayer {
    hp: i32,
    max_hp: i32,
    block: i32,
    energy: i32,
    max_energy: i32,
    stars: i32,
    powers: Vec<CsPower>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsCard {
    id: String,
    base_damage: Option<i32>,
    current_damage: Option<i32>,
    base_block: Option<i32>,
    current_block: Option<i32>,
    magic_number: Option<i32>,
    target_type: Option<String>,
    upgraded: Option<bool>,
    cost: Option<i32>,
    is_playable: Option<bool>,
    is_generated: Option<bool>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsEnemy {
    id: String,
    hp: i32,
    max_hp: i32,
    block: i32,
    is_minion: bool,
    powers: Vec<CsPower>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsPotion {
    id: String,
    name: String,
    can_use: bool,
    target_type: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CsState {
    player: CsPlayer,
    enemies: Vec<CsEnemy>,
    hand: Vec<CsCard>,
    draw_pile: Vec<serde_json::Value>,
    discard_pile: Vec<serde_json::Value>,
    exhaust_pile: Vec<serde_json::Value>,
    potions: Vec<CsPotion>,
    retains_block: bool,
    floor: i32,
    predicted_total_damage: Option<i32>,
    predicted_end_block: Option<i32>,
}

#[derive(Deserialize, Debug)]
struct ActionLog {
    action_idx: usize,
    card: Option<String>,
}

fn convert_powers(cs_powers: Vec<CsPower>) -> Vec<Power> {
    cs_powers.into_iter().map(|p| Power { id: p.id, amount: p.amount }).collect()
}

fn convert_card(c: CsCard) -> Card {
    let tt = match c.target_type.as_deref() {
        Some("AnyEnemy") | Some("SingleEnemy") => TargetType::Single,
        Some("AllEnemies") | Some("AllEnemy") => TargetType::All,
        Some("Self") => TargetType::SelfTarget,
        _ => TargetType::None,
    };
    
    let base_dmg = if let Some(bd) = c.base_damage {
        if bd > 0 { bd } else { c.current_damage.unwrap_or(0) }
    } else {
        c.current_damage.unwrap_or(0)
    };

    let base_blk = if let Some(bb) = c.base_block {
        if bb > 0 { bb } else { c.current_block.unwrap_or(0) }
    } else {
        c.current_block.unwrap_or(0)
    };
    
    Card {
        id: c.id,
        cost: c.cost.unwrap_or(0),
        base_damage: base_dmg,
        base_block: base_blk,
        current_damage: c.current_damage.unwrap_or(base_dmg),
        current_block: c.current_block.unwrap_or(base_blk),
        magic_number: c.magic_number.unwrap_or(0),
        target: tt,
        is_upgraded: c.upgraded.unwrap_or(false),
        is_playable: c.is_playable.unwrap_or(true),
        is_generated: c.is_generated.unwrap_or(false),
    }
}

fn convert_pile(pile: Vec<serde_json::Value>) -> Vec<Card> {
    pile.into_iter().filter_map(|v| {
        let id = if let Some(id) = v.as_str() {
            id.to_string()
        } else if let Some(id) = v.get("id").and_then(|i| i.as_str()) {
            id.to_string()
        } else {
            return None;
        };

        // If it's a full card object, we could convert it properly,
        // but often piles are just IDs. We'll default to a minimal card
        // that's easy to compare if it doesn't have detail.
        Some(Card {
            id,
            cost: v.get("cost").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            base_damage: v.get("baseDamage").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            base_block: v.get("baseBlock").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            current_damage: v.get("currentDamage").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            current_block: v.get("currentBlock").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            magic_number: v.get("magicNumber").and_then(|i| i.as_i64()).map(|i| i as i32).unwrap_or(0),
            target: TargetType::None,
            is_upgraded: v.get("upgraded").and_then(|b| b.as_bool()).unwrap_or(false),
            is_playable: v.get("isPlayable").and_then(|b| b.as_bool()).unwrap_or(true),
            is_generated: v.get("is_generated").and_then(|b| b.as_bool()).unwrap_or(false),
        })
    }).collect()
}

fn convert_to_internal(cs: CsState) -> GameState {
    let mut player = Creature::new( "Player".to_string(), cs.player.max_hp);
    player.cur_hp = cs.player.hp;
    player.block = cs.player.block;
    player.powers = convert_powers(cs.player.powers);
    
    let enemies = cs.enemies.into_iter().map(|e| {
        let mut en = Creature::new(e.id, e.max_hp);
        en.cur_hp = e.hp;
        en.block = e.block;
        en.is_minion = e.is_minion;
        en.powers = convert_powers(e.powers);
        en
    }).collect();
    
    let hand = cs.hand.into_iter().map(convert_card).collect();
    
    let draw_pile = convert_pile(cs.draw_pile);
    let discard_pile = convert_pile(cs.discard_pile);
    let exhaust_pile = convert_pile(cs.exhaust_pile);
    let potions = cs.potions.into_iter().map(|p| battle_simulator::state::Potion {
        id: p.id,
        name: p.name,
        can_use: p.can_use,
        target_type: p.target_type,
    }).collect();

    GameState {
        player,
        enemies,
        hand,
        draw_pile,
        discard_pile,
        exhaust_pile,
        potions,
        energy: cs.player.energy,
        max_energy: cs.player.max_energy,
        stars: cs.player.stars,
        retains_block: cs.retains_block,
        floor: cs.floor,
        predicted_total_damage: cs.predicted_total_damage.unwrap_or(0),
        predicted_end_block: cs.predicted_end_block.unwrap_or(0),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let before_str = fs::read_to_string(&args.state_before)?;
    let action_str = fs::read_to_string(&args.action)?;
    let after_str = fs::read_to_string(&args.state_after)?;

    let cs_before: CsState = serde_json::from_str(&before_str)?;
    let action: ActionLog = serde_json::from_str(&action_str)?;
    let cs_after: CsState = serde_json::from_str(&after_str)?;

    let mut state = convert_to_internal(cs_before);
    let target_state = convert_to_internal(cs_after);

    // Run action
    if action.action_idx < 50 {
        let card_idx = action.action_idx / 5;
        let target_idx = action.action_idx % 5;
        
        println!(">>> Playing card {} at hand index {} targeting enemy index {}", 
                 action.card.as_deref().unwrap_or("Unknown"), card_idx, target_idx);
        state.play_card(card_idx, Some(target_idx));
    } else {
        println!(">>> Action {} not implemented in CLI yet", action.action_idx);
    }

    // Compare
    let sim_json = serde_json::to_string_pretty(&state)?;
    let real_json = serde_json::to_string_pretty(&target_state)?;

    if sim_json == real_json {
        println!("{}", "true".green().bold());
    } else {
        println!("{}", "DISCREPANCY FOUND".red().bold());
        
        let diff = TextDiff::from_lines(&real_json, &sim_json);
        for change in diff.iter_all_changes() {
            match change.tag() {
                ChangeTag::Delete => print!("{}{}", "-".red(), change.value().red()),
                ChangeTag::Insert => print!("{}{}", "+".green(), change.value().green()),
                ChangeTag::Equal => print!(" {}", change.value()),
            }
        }
    }

    Ok(())
}
