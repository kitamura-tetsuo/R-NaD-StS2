# Event dictionary for Slay the Spire 2
# Maps EventId (base.Id.Entry) to structured parameters for each option.

# Parameters:
# hp_loss_pct: Percentage of max HP lost (0.0 to 1.0)
# max_hp_gain: Flat amount of max HP gained
# gold_gain: Flat amount of gold gained
# gold_loss: Flat amount of gold lost
# card_removal_count: Number of cards removed
# card_upgrade_count: Number of cards upgraded
# relic_gain_count: Number of relics gained
# curse_gain_count: Number of curses gained

EVENT_DICT = {
    "ShiningLight": [
        # Option 0: Enter. (Lose HP, Upgrade 2 random cards)
        {"hp_loss_pct": 0.2, "card_upgrade_count": 2},
        # Option 1: Leave.
        {}
    ],
    "BigFish": [
        # Option 0: Banana (Heal 1/3 HP)
        {"hp_gain_pct": 0.33},
        # Option 1: Donut (Max HP +5)
        {"max_hp_gain": 5},
        # Option 2: Box (Relic, Curse)
        {"relic_gain_count": 1, "curse_gain_count": 1}
    ],
    "TheOldBeggar": [
        # Option 0: Give Gold (Remove card)
        {"gold_loss": 75, "card_removal_count": 1},
        # Option 1: Leave
        {}
    ],
    "GoldenIdolEvent": [
        # Option 0: Take (Relic, Trap/Curse)
        {"relic_gain_count": 1, "curse_gain_count": 1},
        # Option 1: Leave
        {}
    ],
    "WorldOfGoop": [
        # Option 0: Reach in (Gold, HP loss)
        {"gold_gain": 75, "hp_loss": 11},
        # Option 1: Leave
        {}
    ],
    "ScrapOoze": [
        # Option 0: Reach in (Relic, HP loss chance - simplified to expected HP loss)
        {"relic_gain_count": 1, "hp_loss": 5},
        # Option 1: Leave
        {}
    ],
}

def get_event_features(event_id, option_idx):
    event = EVENT_DICT.get(event_id)
    if not event or option_idx >= len(event):
        return [0.0] * 10
    
    params = event[option_idx]
    return [
        params.get("hp_loss_pct", 0.0),
        params.get("max_hp_gain", 0.0) / 20.0,
        params.get("gold_gain", 0.0) / 200.0,
        params.get("gold_loss", 0.0) / 200.0,
        params.get("card_removal_count", 0.0) / 2.0,
        params.get("card_upgrade_count", 0.0) / 5.0,
        params.get("relic_gain_count", 0.0) / 2.0,
        params.get("curse_gain_count", 0.0) / 2.0,
        params.get("hp_gain_pct", 0.0),
        params.get("hp_loss", 0.0) / 50.0
    ]
