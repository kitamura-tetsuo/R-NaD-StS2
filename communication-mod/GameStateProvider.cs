using Godot;
using System.Collections.Generic;
using System.Linq;

namespace communication_mod;

public partial class MainFile : Node
{
    private static bool _diagnosed = false;

    private string GetJsonState()
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        if (runState == null) return "{\"type\":\"none\"}";

        if (runState.IsGameOver)
        {
            Logger.Info("[AutoAI] RunState.IsGameOver is true. Reporting game_over state.");
            return "{\"type\":\"game_over\"}";
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
             return System.Text.Json.JsonSerializer.Serialize(new { type = "unknown", error = "currentRoom is null" }, JsonOptions);
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
                rewards = rewards,
                has_open_potion_slots = hasOpenPotionSlots,
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

            return System.Text.Json.JsonSerializer.Serialize(new { type = "card_reward", cards = cards, buttons = buttons }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen gos)
        {
            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "game_over",
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
                    relics = relics
                }, JsonOptions);
            }
        }

        // 2. Room Logic
        if (currentRoom == null) return "{\"type\":\"none\"}";

        if (currentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom combatRoom)
        {
            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            if (cm == null || !cm.IsInProgress || !cm.IsPlayPhase) return "{\"type\":\"combat_waiting\"}";

            // Check if hand is in selection mode (e.g., Armaments)
            var hand = MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Instance;
            if (hand != null)
            {
                var upgradePreviewField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_upgradePreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var upgradePreview = upgradePreviewField?.GetValue(hand) as CanvasItem;
                
                var simplePreviewField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectedHandCardContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var simplePreview = simplePreviewField?.GetValue(hand) as CanvasItem;

                var confirmBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectModeConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var confirmBtn = confirmBtnField?.GetValue(hand) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;

                // Use the explicit Mode enum to distinguish between normal play and selection screens
                // Mode 2 = SimpleSelect, Mode 3 = UpgradeSelect (based on decompiled enum)
                bool isSelectionMode = hand.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.SimpleSelect || 
                                     hand.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.UpgradeSelect;

                if (isSelectionMode || hand.IsInCardSelection)
                {
                    // Logger.Info($"[AutoAI] hand_selection detected via Mode={hand.CurrentMode} IsInCardSelection={hand.IsInCardSelection}");
                    var cards = new List<object>();
                    var activeHolders = hand.ActiveHolders;
                    for (int i = 0; i < activeHolders.Count; i++)
                    {
                        var holder = activeHolders[i];
                        if (holder.CardNode != null)
                        {
                            cards.Add(new { index = i, name = holder.CardNode.Model?.Title ?? "Unknown" });
                        }
                    }

                    bool isConfirming = confirmBtn != null && confirmBtn.IsVisibleInTree() && confirmBtn.IsEnabled;
                    
                    if (confirmBtn != null && confirmBtn.IsVisibleInTree())
                    {
                        var selectedCardsField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectedCards", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var selectedCards = selectedCardsField?.GetValue(hand) as System.Collections.IEnumerable;
                        int count = 0;
                        if (selectedCards != null) foreach (var c in selectedCards) count++;

                        // Logger.Info($"[AutoAI] hand confirmBtn: Enabled={confirmBtn.IsEnabled} Peeking={hand.PeekButton.IsPeeking} count={count}");
                    }

                    return System.Text.Json.JsonSerializer.Serialize(new
                    {
                        type = "hand_selection",
                        cards = cards,
                        is_confirming = isConfirming,
                        mode = hand.CurrentMode.ToString()
                    }, JsonOptions);
                }
            }

            if (cm.PlayerActionsDisabled) return "{\"type\":\"combat_waiting\"}";

            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            var pState = player?.PlayerCombatState;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "combat",
                player = new
                {
                    hp = player?.Creature.CurrentHp ?? 0,
                    maxHp = player?.Creature.MaxHp ?? 0,
                    block = player?.Creature.Block ?? 0,
                    energy = pState?.Energy ?? 0
                },
                hand = pState?.Hand.Cards.Select(c => new
                {
                    id = c.Id.Entry,
                    name = c.Title,
                    isPlayable = c.CanPlay()
                }).ToList(),
                potions = player?.PotionSlots.Select((p, i) => new
                {
                    index = i,
                    id = p?.Id.Entry ?? "empty",
                    name = p?.Title.GetRawText() ?? "Empty Slot",
                    canUse = p != null && p.PassesCustomUsabilityCheck && (p.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime || (p.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly && MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress))
                }).ToList(),
                enemies = combatRoom.Enemies.Where(e => e.IsAlive).Select(e => new
                {
                    id = e.ModelId.Entry,
                    name = e.Name,
                    hp = e.CurrentHp
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
                has_chest = hasChest,
                can_proceed = canProceed
            }, JsonOptions);
        }

        Logger.Info($"[AutoAI] Reporting unknown state for room: {currentRoom.GetType().Name}");
        return System.Text.Json.JsonSerializer.Serialize(new { type = "unknown", room = currentRoom.GetType().Name }, JsonOptions);
    }

    private string GetMapJson(MegaCrit.Sts2.Core.Runs.RunState runState)
    {
        var currentPos = runState.CurrentMapCoord;
        var nextNodesData = new List<object>();

        // New Logic: Try to use NMapScreen's internal state to find Travelable nodes
        var mapScreen = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance;
        bool foundViaScreen = false;

        if (mapScreen != null)
        {
            try
            {
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen).GetField("_mapPointDictionary", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var dict = field?.GetValue(mapScreen) as System.Collections.IDictionary;

                if (dict != null)
                {
                    foreach (System.Collections.DictionaryEntry entry in dict)
                    {
                        var nPoint = entry.Value as MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapPoint;
                        if (nPoint != null)
                        {
                            // Check the private IsTravelable property or State
                            // According to decompilation, IsTravelable checks Debug state OR (IsTravelEnabled && State == Travelable)
                            // We can use reflection to get IsTravelable
                            var prop = typeof(MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapPoint).GetProperty("IsTravelable", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                            bool isTravelable = (bool)(prop?.GetValue(nPoint) ?? false);

                            if (isTravelable)
                            {
                                var p = nPoint.Point;
                                nextNodesData.Add(new { row = p.coord.row, col = p.coord.col, type = p.PointType.ToString() });
                                foundViaScreen = true;
                            }
                        }
                    }
                }
            }
            catch (System.Exception ex)
            {
                Logger.Error($"[AutoAI] Error accessing _mapPointDictionary via reflection: {ex.Message}");
            }
        }

        if (!foundViaScreen)
        {
            Logger.Info("[AutoAI] MapScreen not available or empty. Falling back to Children-based next_nodes logic.");
            System.Collections.Generic.IEnumerable<MegaCrit.Sts2.Core.Map.MapPoint> nextPoints;
            if (currentPos.HasValue)
            {
                var point = runState.Map.GetPoint(currentPos.Value);
                nextPoints = point?.Children ?? new System.Collections.Generic.HashSet<MegaCrit.Sts2.Core.Map.MapPoint>();
            }
            else
            {
                nextPoints = runState.Map.StartingMapPoint?.Children ?? new System.Collections.Generic.HashSet<MegaCrit.Sts2.Core.Map.MapPoint>();
                if (!nextPoints.Any())
                {
                    nextPoints = runState.Map.GetPointsInRow(0);
                }
            }
            foreach (var p in nextPoints)
            {
                nextNodesData.Add(new { row = p.coord.row, col = p.coord.col, type = p.PointType.ToString() });
            }
        }

        return System.Text.Json.JsonSerializer.Serialize(new
        {
            type = "map",
            current_pos = currentPos.HasValue ? new { row = currentPos.Value.row, col = currentPos.Value.col } : null,
            next_nodes = nextNodesData
        }, JsonOptions);
    }
}
