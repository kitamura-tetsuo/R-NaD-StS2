using Godot;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;

namespace communication_mod;

public partial class MainFile : Node
{
    private static bool _diagnosed = false;

    private string GetJsonState()
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        if (runState == null) return "{\"type\":\"none\", \"floor\": 0}";

        if (runState.IsGameOver)
        {
            Logger.Info("[AutoAI] RunState.IsGameOver is true. Reporting game_over state.");
            return System.Text.Json.JsonSerializer.Serialize(new { type = "game_over", floor = runState.TotalFloor }, JsonOptions);
        }

        var currentRoom = runState.CurrentRoom;

        var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
        var topOverlay = (overlayStack?.ScreenCount ?? 0) > 0 ? overlayStack.Peek() : null;

        // Prioritize Map if Screen is actually open OR if we are in a run but have no room yet (initial act start)
        bool mapScreenExists = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance != null;
        bool mapScreenOpen = mapScreenExists && MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance.IsOpen;

        if (mapScreenExists && (mapScreenOpen || currentRoom == null))
        {
            // Only report map if no major overlays are on top (except Rewards which we handle specially)
            bool isRewardsOrNothing = topOverlay == null || topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen;

            if (isRewardsOrNothing)
            {
               // Still check if rewards has items
               bool hasRewards = false;
               if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen rewardsScreen)
               {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen).GetField("_rewardButtons", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var buttons = field?.GetValue(rewardsScreen) as System.Collections.IEnumerable;
                    if (buttons != null)
                    {
                        foreach (var btn in buttons)
                        {
                            var rewardProp = btn.GetType().GetProperty("Reward");
                            if (rewardProp?.GetValue(btn) != null) { hasRewards = true; break; }
                        }
                    }
               }

               if (!hasRewards)
               {
                   Logger.Info("[AutoAI] Map Screen is open and no rewards pending. Reporting map state.");
                   return GetMapJson(runState);
               }
            }
        }

        if (currentRoom == null)
        {
             Logger.Info($"[AutoAI] currentRoom is null. mapScreenExists={mapScreenExists}, mapScreenOpen={mapScreenOpen}");
             return System.Text.Json.JsonSerializer.Serialize(new { type = "unknown", floor = runState.TotalFloor, error = "currentRoom is null" }, JsonOptions);
        }

        if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen rs)
        {
            var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen).GetField("_rewardButtons", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var buttons = field?.GetValue(rs) as System.Collections.IEnumerable;

            var rewards = new List<object>();
            bool hasRewards = false;
            if (buttons != null)
            {
                int i = 0;
                foreach (var btn in buttons)
                {
                    var rewardProp = btn.GetType().GetProperty("Reward");
                    var reward = rewardProp?.GetValue(btn) as MegaCrit.Sts2.Core.Rewards.Reward;
                    if (reward != null)
                    {
                        var rewardType = reward.GetType().Name;
                        rewards.Add(new { index = i, type = rewardType, description = reward.Description.GetRawText() });
                        hasRewards = true;
                    }
                    i++;
                }
            }

            // If map is open and we have no rewards left, report map instead of the overlay
            if (!hasRewards && MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance != null && MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance.IsOpen)
            {
                Logger.Info("[AutoAI] RewardsScreen is empty and Map is open. Reporting map state.");
                return GetMapJson(runState);
            }

            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            bool hasOpenPotionSlots = player?.HasOpenPotionSlots ?? false;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "rewards",
                floor = runState.TotalFloor,
                rewards = rewards,
                has_open_potion_slots = hasOpenPotionSlots,
                relics = player?.Relics.Select(r => r.Model.Id.Entry).ToList() ?? new List<string>(),
                can_proceed = MegaCrit.Sts2.Core.Hooks.Hook.ShouldProceedToNextMapPoint(runState)
            }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen cardRewardScreen)
        {
            var cards = new List<object>();
            var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(cardRewardScreen);
            for (int i = 0; i < holders.Count; i++)
            {
                cards.Add(new { index = i, name = holders[i].CardModel?.Title ?? "Unknown" });
            }

            var buttons = new List<object>();
            var altButtons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton>(cardRewardScreen);
            if (altButtons.Count > 0 && !_diagnosed) {
                var btn = altButtons[0];
                Logger.Info($"[AutoAI] Diagnosing button fields for {btn.GetType().FullName}");
                foreach (var f in btn.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)) {
                   Logger.Info($"[AutoAI] Field: {f.Name} Value: {f.GetValue(btn)}");
                }
                _diagnosed = true;
            }

            for (int i = 0; i < altButtons.Count; i++)
            {
                var btn = altButtons[i];
                string btnName = "Unknown";

                // Try to get _optionName field
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton).GetField("_optionName", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (field != null) {
                    btnName = field.GetValue(btn) as string ?? "NullName";
                } else {
                    // Try to get from label child
                    var label = btn.GetNodeOrNull<MegaCrit.Sts2.addons.mega_text.MegaLabel>("Label");
                    if (label != null) {
                        btnName = label.Text;
                    }
                }

                if (string.IsNullOrEmpty(btnName) || btnName == "NullName") btnName = btn.Name;
                buttons.Add(new { index = i, name = btnName });
            }

            if (holders.Count == 0 && altButtons.Count == 0)
            {
                Logger.Info($"[AutoAI] CardRewardScreen empty! Total children: {cardRewardScreen.GetChildCount()}");
                foreach (var child in cardRewardScreen.GetChildren())
                {
                    Logger.Info($"[AutoAI] Child node: {child.GetType().FullName} Name: {child.Name}");
                }
            }

            return System.Text.Json.JsonSerializer.Serialize(new { type = "card_reward", floor = runState.TotalFloor, cards = cards, buttons = buttons }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen gos)
        {
            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "game_over",
                floor = runState.TotalFloor,
                victory = runState?.CurrentRoom?.IsVictoryRoom ?? false
            }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen chooseScreen)
        {
            var cards = new List<object>();
            var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(chooseScreen);
            for (int i = 0; i < holders.Count; i++)
            {
                cards.Add(new { index = i, name = holders[i].CardModel?.Title ?? "Unknown" });
            }

            var skipBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen).GetField("_skipButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var skipBtn = skipBtnField?.GetValue(chooseScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
            bool canSkip = skipBtn != null && skipBtn.IsEnabled && skipBtn.Visible;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = "choose_a_card",
                floor = runState.TotalFloor,
                cards = cards,
                can_skip = canSkip
            }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen gridSelection)
        {
            Logger.Info($"[AutoAI] Grid Selection detected: {gridSelection.GetType().FullName}");
            var cards = new List<object>();
            var gridField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen).GetField("_grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var grid = gridField?.GetValue(gridSelection) as MegaCrit.Sts2.Core.Nodes.Cards.NCardGrid;

            if (grid != null)
            {
                var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(grid);
                for (int i = 0; i < holders.Count; i++)
                {
                    cards.Add(new { index = i, name = holders[i].CardModel?.Title ?? "Unknown" });
                }
            }

            bool isConfirming = false;
            if (gridSelection is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen upgradeScreen)
            {
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_upgradeSinglePreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var single = field?.GetValue(upgradeScreen) as CanvasItem;
                var fieldMulti = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_upgradeMultiPreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var multi = fieldMulti?.GetValue(upgradeScreen) as CanvasItem;
                isConfirming = (single != null && single.Visible) || (multi != null && multi.Visible);
            }
            else if (gridSelection is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen transformScreen)
            {
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_previewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var preview = field?.GetValue(transformScreen) as CanvasItem;
                isConfirming = preview != null && preview.Visible;
            }
            else if (gridSelection is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen cardSelectScreen)
            {
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen).GetField("_previewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var preview = field?.GetValue(cardSelectScreen) as CanvasItem;
                isConfirming = preview != null && preview.Visible;
            }
            else if (gridSelection is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen enchantScreen)
            {
                var singleField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_enchantSinglePreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var multiField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_enchantMultiPreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                
                var singlePreview = singleField?.GetValue(enchantScreen) as CanvasItem;
                var multiPreview = multiField?.GetValue(enchantScreen) as CanvasItem;
                
                isConfirming = (singlePreview != null && singlePreview.Visible) || (multiPreview != null && multiPreview.Visible);
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = gridSelection.GetType().Name,
                floor = runState.TotalFloor,
                cards = cards,
                is_confirming = isConfirming
            }, JsonOptions);
        }

        // 1.5. Check for global screens that aren't on overlay stack
        var relicCollection = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicCollection>(GetTree().Root).FirstOrDefault(c => ((CanvasItem)c).Visible);
        if (relicCollection != null)
        {
            var relics = new List<object>();
            var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicHolder>(relicCollection);
            for (int i = 0; i < holders.Count; i++)
            {
                var holder = holders[i];
                if (((CanvasItem)holder).Visible)
                {
                    relics.Add(new { index = holder.Index, name = holder.Relic?.Model?.Id.Entry ?? "Unknown" });
                }
            }

            if (relics.Count > 0)
            {
                return System.Text.Json.JsonSerializer.Serialize(new
                {
                    type = "treasure_relics",
                    floor = runState.TotalFloor,
                    relics = relics
                }, JsonOptions);
            }
        }

        // 2. Room Logic
        if (currentRoom == null) return "{\"type\":\"none\"}";

        if (currentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom combatRoom)
        {
            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            bool queueProcessing = rm.ActionQueueSet != null && !rm.ActionQueueSet.IsEmpty;

            bool combatBusy = cm != null && cm.IsInProgress && (!cm.IsPlayPhase || cm.PlayerActionsDisabled);
            if (combatBusy || queueProcessing) 
            {
                if (queueProcessing) {
                    Logger.Info("[AutoAI] combat_waiting: ActionQueue is not empty. Waiting for execution.");
                }
                return "{\"type\":\"combat_waiting\"}";
            }

            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            var pState = player?.PlayerCombatState;

            // Diagnostic: Log card properties once
            if (!_diagnosed && pState?.Hand.Cards.Count > 0)
            {
                var card = pState.Hand.Cards[0];
                Logger.Info($"[AutoAI] Diagnosing card properties for {card.GetType().FullName}");
                foreach (var prop in card.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                {
                    try {
                        Logger.Info($"[AutoAI] Property: {prop.Name} Value: {prop.GetValue(card)}");
                    } catch {}
                }
                _diagnosed = true;
            }

            // Diagnostic: Log creature (player) properties once
            if (!_diagnosed && player != null)
            {
                Logger.Info($"[AutoAI] Diagnosing creature properties for {player.Creature.GetType().FullName}");
                foreach (var prop in player.Creature.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                {
                    try {
                        Logger.Info($"[AutoAI] Property: {prop.Name}");
                    } catch {}
                }
            }

            // Check if hand is in selection mode (e.g., Armaments, Grid selection in combat)
            var handNode = MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Instance;
            if (handNode != null)
            {
                var confirmBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectModeConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var confirmBtn = confirmBtnField?.GetValue(handNode) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;

                bool isSelectionMode = handNode.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.SimpleSelect || 
                                     handNode.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.UpgradeSelect;

                if (isSelectionMode || handNode.IsInCardSelection)
                {
                    var cards = new List<object>();
                    var activeHolders = handNode.ActiveHolders;
                    for (int i = 0; i < activeHolders.Count; i++)
                    {
                        var holder = activeHolders[i];
                        if (holder.CardNode != null)
                        {
                            cards.Add(new { index = i, name = holder.CardNode.Model?.Title ?? "Unknown" });
                        }
                    }

                    bool isConfirming = confirmBtn != null && confirmBtn.IsVisibleInTree() && confirmBtn.IsEnabled;
                    
                    return System.Text.Json.JsonSerializer.Serialize(new
                    {
                        type = "hand_selection",
                        floor = runState.TotalFloor,
                        cards = cards,
                        is_confirming = isConfirming,
                        mode = handNode.CurrentMode.ToString()
                    }, JsonOptions);
                }
            }

            if (cm.PlayerActionsDisabled) return "{\"type\":\"combat_waiting\"}";

            var combatNode = MegaCrit.Sts2.Core.Nodes.Rooms.NCombatRoom.Instance;
            bool canProceed = combatNode?.ProceedButton?.IsEnabled ?? false;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "combat",
                floor = runState.TotalFloor,
                can_proceed = canProceed,
                player = new
                {
                    hp = player?.Creature.CurrentHp ?? 0,
                    maxHp = player?.Creature.MaxHp ?? 0,
                    block = player?.Creature.Block ?? 0,
                    energy = pState?.Energy ?? 0,
                    maxEnergy = pState?.MaxEnergy ?? 0,
                    stars = pState?.Stars ?? 0,
                    drawPile = pState?.DrawPile.Cards.Select(c => c.Id.Entry).ToList() ?? new List<string>(),
                    discardPile = pState?.DiscardPile.Cards.Select(c => c.Id.Entry).ToList() ?? new List<string>(),
                    exhaustPile = pState?.ExhaustPile.Cards.Select(c => c.Id.Entry).ToList() ?? new List<string>(),
                    masterDeck = GetPropValue<string>(runState, "MasterDeck", GetPropValue<string>(runState, "Deck", GetPropValue<string>(player, "MasterDeck", "[]"))),
                    relics = player?.Relics.Select(r => r.Model.Id.Entry).ToList() ?? new List<string>(),
                    powers = player?.Creature.Powers.Select(p => new { id = p.Model.Id.Entry, amount = p.Amount }).ToList() ?? new List<object>()
                },
                hand = pState?.Hand.Cards.Select(c => {
                    var dynamicVars = c.DynamicVars;
                    var firstEnemy = combatRoom.Enemies.FirstOrDefault(e => e.IsAlive);
                    c.UpdateDynamicVarPreview(MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.Normal, firstEnemy, dynamicVars);
                    
                    int curDamage = 0;
                    int curBlock = 0;
                    if (dynamicVars.ContainsKey("Damage")) curDamage = (int)dynamicVars["Damage"].PreviewValue;
                    if (dynamicVars.ContainsKey("CalculatedBlock")) curBlock = (int)dynamicVars["CalculatedBlock"].PreviewValue;
                    else if (dynamicVars.ContainsKey("Block")) curBlock = (int)dynamicVars["Block"].PreviewValue;

                    return new
                    {
                        id = c.Id.Entry,
                        name = c.Title,
                        isPlayable = c.CanPlay(),
                        targetType = c.TargetType.ToString(),
                        baseDamage = GetPropValue(c, "BaseDamage", 0),
                        baseBlock = GetPropValue(c, "BaseBlock", 0),
                        magicNumber = GetPropValue(c, "MagicNumber", 0),
                        cost = GetPropValue(c, "BaseCost", 0),
                        currentDamage = curDamage,
                        currentBlock = curBlock
                    };
                }).ToList(),
                potions = player?.PotionSlots.Select((p, i) => new
                {
                    index = i,
                    id = p?.Id.Entry ?? "empty",
                    name = p?.Title.GetRawText() ?? "Empty Slot",
                    canUse = p != null && p.PassesCustomUsabilityCheck && (p.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime || (p.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly && MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress)),
                    targetType = p?.TargetType.ToString() ?? "None"
                }).ToList(),
                enemies = combatRoom.Enemies.Where(e => e.IsAlive).Select(e => {
                    var mModel = e.Monster;
                    var intents = new List<object>();
                    if (mModel != null) {
                        foreach (var intent in mModel.NextMove.Intents) {
                            int damage = 0;
                            int repeats = 0;
                            if (intent is MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent attackIntent) {
                                damage = attackIntent.GetSingleDamage(combatRoom.CombatState.PlayerCreatures, e);
                                repeats = attackIntent.Repeats;
                            }
                            intents.Add(new {
                                type = intent.IntentType.ToString(),
                                damage = damage,
                                repeats = repeats
                            });
                        }
                    }
                    return new
                    {
                        id = e.ModelId.Entry,
                        name = e.Name,
                        hp = e.CurrentHp,
                        maxHp = e.MaxHp,
                        block = e.Block,
                        powers = e.Powers.Select(p => new { id = p.Model.Id.Entry, amount = p.Amount }).ToList() ?? new List<object>(),
                        intents = intents
                    };
                }).ToList()
            }, JsonOptions);
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.MapRoom || (MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance?.IsOpen ?? false))
        {
            return GetMapJson(runState);
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom er)
        {
            var ev = er.LocalMutableEvent;
            if (ev == null) return "{\"type\":\"event_none\"}";

            var options = new List<object>();
            var modelOptions = ev.CurrentOptions;

            if (modelOptions.Count > 0) {
                for (int i = 0; i < modelOptions.Count; i++) {
                    options.Add(new {
                        index = i,
                        title = modelOptions[i].Title.GetRawText(),
                        is_locked = modelOptions[i].IsLocked
                    });
                }
            } else {
                // Fallback to UI buttons (useful for PROCEED buttons or when model is out of sync)
                var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom>(GetTree().Root)
                    .SelectMany(r => FindNodesByType<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>(r))
                    .OrderBy(b => b.GlobalPosition.Y) // Usually top to bottom
                    .ToList();

                if (buttons.Count > 0) {
                    for (int i = 0; i < buttons.Count; i++) {
                        var opt = buttons[i].Option;
                        options.Add(new {
                            index = i,
                            title = opt.Title.GetRawText() ?? "Proceed",
                            is_locked = opt.IsLocked,
                            is_ui_fallback = true
                        });
                    }
                }
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "event",
                floor = runState.TotalFloor,
                id = ev.Id.Entry,
                title = ev.Title.GetRawText(),
                options = options
            }, JsonOptions);
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.RestSiteRoom rsr)
        {
            var options = rsr.Options;
            var optionsData = new List<object>();
            
            if (options.Count > 0)
            {
                for (int i = 0; i < options.Count; i++)
                {
                    optionsData.Add(new
                    {
                        index = i,
                        title = options[i].Title.GetRawText(),
                        is_enabled = options[i].IsEnabled
                    });
                }
            }
            else
            {
                // UI Fallback for Rest Site Buttons
                var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton>(GetTree().Root)
                    .Where(b => b.Visible)
                    .OrderBy(b => b.GlobalPosition.X)
                    .ToList();
                
                if (buttons.Count > 0)
                {
                    Logger.Info($"[AutoAI] Found {buttons.Count} Rest Site buttons via UI fallback.");
                    for (int i = 0; i < buttons.Count; i++)
                    {
                        var opt = buttons[i].Option;
                        optionsData.Add(new
                        {
                            index = i,
                            title = opt.Title.GetRawText(),
                            is_enabled = opt.IsEnabled,
                            is_ui_fallback = true
                        });
                    }
                }
            }

            var restSiteNode = MegaCrit.Sts2.Core.Nodes.Rooms.NRestSiteRoom.Instance;
            bool canProceed = restSiteNode?.ProceedButton?.IsEnabled ?? false;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "rest_site",
                floor = runState.TotalFloor,
                options = optionsData,
                can_proceed = canProceed
            }, JsonOptions);
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.MerchantRoom)
        {
            var merchantRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
            bool canProceed = merchantRoom?.ProceedButton?.IsEnabled ?? false;
            
            var items = new List<object>();
            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            int gold = player?.Gold ?? 0;

            var inventoryNode = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory>(GetTree().Root).FirstOrDefault(n => n.Visible);
            if (inventoryNode != null)
            {
                var slots = inventoryNode.GetAllSlots();
                foreach (var slot in slots)
                {
                    if (slot.Entry != null && slot.Entry.IsStocked)
                    {
                        items.Add(new {
                            index = slots.ToList().IndexOf(slot),
                            name = slot.Entry.GetType().Name, // Fallback name
                            cost = slot.Entry.Cost,
                            canAfford = slot.Entry.EnoughGold
                        });
                    }
                }
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "shop",
                floor = runState.TotalFloor,
                gold = gold,
                items = items,
                can_proceed = canProceed
            }, JsonOptions);
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.TreasureRoom tr)
        {
            var treasureRoomNode = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom>(GetTree().Root).FirstOrDefault();
            bool hasChest = false;
            bool canProceed = false;

            if (treasureRoomNode != null)
            {
                var chestBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom).GetField("_chestButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var chestBtn = chestBtnField?.GetValue(treasureRoomNode) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                hasChest = chestBtn != null && chestBtn.Visible && chestBtn.IsEnabled;
                canProceed = treasureRoomNode.ProceedButton?.IsEnabled ?? false;
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "treasure",
                floor = runState.TotalFloor,
                has_chest = hasChest,
                can_proceed = canProceed
            }, JsonOptions);
        }

        Logger.Info($"[AutoAI] Reporting unknown state for room: {currentRoom.GetType().Name}");
        return System.Text.Json.JsonSerializer.Serialize(new { type = "unknown", floor = runState.TotalFloor, room = currentRoom.GetType().Name }, JsonOptions);
    }

    private string GetMapJson(MegaCrit.Sts2.Core.Runs.RunState runState)
    {
        var currentPos = runState.CurrentMapCoord;
        var nodes = new List<object>();
        var edges = new List<object>();

        // Collect all nodes and their outgoing edges
        // The map in StS2 usually has around 15-20 rows.
        for (int row = 0; row < 30; row++) 
        {
            var pointsInRow = runState.Map.GetPointsInRow(row);
            if (pointsInRow == null || !pointsInRow.Any()) 
            {
                if (row > 5) break; 
                continue;
            }

            foreach (var p in pointsInRow)
            {
                nodes.Add(new { 
                    row = p.coord.row, 
                    col = p.coord.col, 
                    type = p.PointType.ToString() 
                });

                if (p.Children != null)
                {
                    foreach (var child in p.Children)
                    {
                        edges.Add(new {
                            src_row = p.coord.row,
                            src_col = p.coord.col,
                            dst_row = child.coord.row,
                            dst_col = child.coord.col
                        });
                    }
                }
            }
        }

        // Add Boss nodes if they aren't in GetPointsInRow
        if (runState.Map.BossMapPoint != null) {
            var b = runState.Map.BossMapPoint;
            nodes.Add(new { row = b.coord.row, col = b.coord.col, type = b.PointType.ToString() });
        }

        // Try to get Boss information
        string bossId = "Unknown";
        try {
            var currentAct = GetPropValue<object>(runState, "CurrentAct", null);
            if (currentAct != null) {
                var bossEncounter = GetPropValue<object>(currentAct, "BossEncounter", null);
                if (bossEncounter != null) {
                    var idObj = GetPropValue<object>(bossEncounter, "Id", null);
                    if (idObj != null) {
                        bossId = GetPropValue<string>(idObj, "Entry", bossId);
                    }
                }
            }
        } catch (System.Exception ex) {
            Logger.Error($"[AutoAI] Error getting BossId: {ex.Message}");
        }

        return System.Text.Json.JsonSerializer.Serialize(new
        {
            type = "map",
            floor = runState.TotalFloor,
            current_pos = currentPos.HasValue ? new { row = currentPos.Value.row, col = currentPos.Value.col } : null,
            nodes = nodes,
            edges = edges,
            boss = bossId
        }, JsonOptions);
    }

    private T GetPropValue<T>(object obj, string propName, T defaultValue)
    {
        if (obj == null) return defaultValue;
        var prop = obj.GetType().GetProperty(propName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (prop != null)
        {
            try {
                var val = prop.GetValue(obj);
                if (val is T t) return t;
                if (typeof(T) == typeof(int) && val is float f) return (T)(object)(int)f;
                if (typeof(T) == typeof(string) && val != null) return (T)(object)val.ToString();
            } catch {}
        }
        return defaultValue;
    }
}
