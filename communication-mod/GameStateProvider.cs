using Godot;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;

namespace communication_mod;

public partial class MainFile : Node
{
    private static bool _diagnosed = false;
    private static string _lastMapJson = "";
    private static int _lastMapFloor = -1;
    private static string _lastMapBoss = "";
    private static (int row, int col)? _lastMapPos = null;

    // Combat Prediction Verification
    private static int _lastPredictedDamage = 0;
    private static int _lastPredictedBlock = 0;
    private static int _lastPlayerHp = 0;
    private static int _lastFloor = -1;
    private static bool _lastRetainsBlock = false;
    private static bool _waitingForVerification = false;

    private void VerifyCombatPrediction(MegaCrit.Sts2.Core.Entities.Players.Player player, int currentFloor)
    {
        if (!_waitingForVerification) return;
        _waitingForVerification = false;

        if (currentFloor != _lastFloor) return;

        int currentHp = player?.Creature.CurrentHp ?? 0;
        int currentBlock = player?.Creature.Block ?? 0;

        int actualHpLoss = _lastPlayerHp - currentHp;
        int expectedHpLoss = Math.Max(0, _lastPredictedDamage - _lastPredictedBlock);

        bool mismatch = false;
        string reason = "";

        if (actualHpLoss != expectedHpLoss)
        {
            mismatch = true;
            reason += $"HP Loss Mismatch: Actual {actualHpLoss}, Expected {expectedHpLoss}. ";
        }

        if (_lastRetainsBlock)
        {
            int expectedRemainingBlock = Math.Max(0, _lastPredictedBlock - _lastPredictedDamage);
            if (currentBlock < expectedRemainingBlock)
            {
                mismatch = true;
                reason += $"Retained Block Mismatch: Actual {currentBlock}, Expected {expectedRemainingBlock}. ";
            }
        }

        if (mismatch)
        {
            string logMsg = $"\n[ERROR] Combat prediction verification FAILED!\n" +
                            $"Reason: {reason}\n" +
                            $"Predictions: Damage={_lastPredictedDamage}, EndBlock={_lastPredictedBlock}, Retains={_lastRetainsBlock}\n" +
                            $"Previous Turn: HP={_lastPlayerHp}\n" +
                            $"Current Turn: HP={currentHp}, Block={currentBlock}\n";
            Logger.Error(logMsg);
            GD.PrintErr(logMsg);
            // System.Environment.Exit(1); // Made non-fatal to avoid "freezing" the inference run
        }
        else
        {
            Logger.Info($"[VERIFIED] Combat prediction matched! HP Loss: {actualHpLoss}, Current Block: {currentBlock}");
        }
    }

    private string GetJsonState()
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        if (runState == null) {
            // If runState is null, it means we are likely in the main menu or transitioning.
            // For the AI bridge, reporting this as game_over allows for a clean reset of trajectories.
            return "{\"type\":\"game_over\", \"floor\": 0, \"victory\": false, \"reason\": \"main_menu\"}";
        }

        string currentSeed = runState.Rng.StringSeed;

        var currentRoom = runState.CurrentRoom;
        var topOverlay = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance?.Peek();
        if (topOverlay != null) Logger.Info($"[AutoAI] Top Overlay: {topOverlay.GetType().FullName}");

        // 1.0 Manual HP Check for Death Detection (Prioritize this for GYM mode and fast death detection)
        var localPlayer = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        if (localPlayer != null && localPlayer.Creature != null && localPlayer.Creature.CurrentHp <= 0)
        {
            Logger.Info("[AutoAI] Manual HP check: Player HP is 0. Reporting game_over state.");
            return System.Text.Json.JsonSerializer.Serialize(new { type = "game_over", floor = runState.TotalFloor, seed = currentSeed, is_gym = _gymMode, victory = false }, JsonOptions);
        }

        // 1. Game Over Check (Screen or State)
        if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen gos)
        {
            bool isVictory = (currentRoom?.IsVictoryRoom ?? false) || !runState.IsGameOver;
            Logger.Info($"[AutoAI] GameOverScreen detected. Victory: {isVictory}");
            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "game_over",
                floor = runState.TotalFloor,
                seed = currentSeed,
                is_gym = _gymMode,
                victory = isVictory
            }, JsonOptions);
        }

        if (runState.IsGameOver)
        {
            Logger.Info("[AutoAI] RunState.IsGameOver is true (Death). Reporting game_over state.");
            return System.Text.Json.JsonSerializer.Serialize(new { type = "game_over", floor = runState.TotalFloor, seed = currentSeed, is_gym = _gymMode, victory = false }, JsonOptions);
        }

        // Prioritize Map if Screen is actually open OR if we are in a run but have no room yet (initial act start)
        bool mapScreenExists = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance != null;
        bool mapScreenOpen = mapScreenExists && MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance.IsOpen;

        if (mapScreenExists && mapScreenOpen)
        {
            // Only report map if no major overlays are on top (except Rewards/Intro which we handle specially)
            bool isRewardsOrIntroOrNothing = topOverlay == null || 
                                             topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen ||
                                             topOverlay.GetType().FullName.Contains("IntroScreen");
            
            if (isRewardsOrIntroOrNothing)
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

            // Check if proceed button is enabled via reflection
            bool proceedBtnEnabled = false;
            try {
                var rsType = typeof(MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen);
                var proceedBtnField = rsType.GetField("_proceedButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var proceedBtn = proceedBtnField?.GetValue(rs) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                proceedBtnEnabled = proceedBtn != null && proceedBtn.IsEnabled && proceedBtn.IsVisibleInTree();
            } catch (Exception ex) {
                Logger.Error($"[AutoAI] Error getting rewards proceed button status: {ex.Message}");
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "rewards",
                floor = runState.TotalFloor,
                seed = currentSeed,
                is_gym = _gymMode,
                rewards = rewards,
                has_open_potion_slots = hasOpenPotionSlots,
                relics = player?.Relics.Select(r => r.Id.Entry).ToList() ?? new List<string>(),
                can_proceed = proceedBtnEnabled || MegaCrit.Sts2.Core.Hooks.Hook.ShouldProceedToNextMapPoint(runState)
            }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen cardRewardScreen)
        {
            var cards = new List<object>();
            // Optimized search: limit to screen children
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

            return System.Text.Json.JsonSerializer.Serialize(new { type = "card_reward", floor = runState.TotalFloor, seed = currentSeed, is_gym = _gymMode, cards = cards, buttons = buttons }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen chooseScreen)
        {
            var cards = new List<object>();
            var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(chooseScreen);
            for (int i = 0; i < holders.Count; i++)
            {
                var model = holders[i].CardModel;
                cards.Add(new { 
                    index = i, 
                    name = model?.Title ?? "Unknown",
                    id = model?.Id.Entry ?? "unknown",
                    upgraded = model != null && (GetPropValue(model, "IsUpgraded", false) || GetPropValue(model, "TimesUpgraded", 0) > 0),
                    cost = model != null ? GetPropValue(model, "BaseCost", 0) : 0
                });
            }

            var skipBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen).GetField("_skipButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var skipBtn = skipBtnField?.GetValue(chooseScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
            bool canSkip = skipBtn != null && skipBtn.IsEnabled && skipBtn.Visible;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = "choose_a_card",
                floor = runState.TotalFloor,
                seed = currentSeed,
                is_gym = _gymMode,
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
                var selectedCardsField = gridSelection.GetType().GetField("_selectedCards", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var selectedCards = selectedCardsField?.GetValue(gridSelection) as IEnumerable<MegaCrit.Sts2.Core.Models.CardModel>;
                var highlightedCardsField = typeof(MegaCrit.Sts2.Core.Nodes.Cards.NCardGrid).GetField("_highlightedCards", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var highlightedCards = highlightedCardsField?.GetValue(grid) as IEnumerable<MegaCrit.Sts2.Core.Models.CardModel>;

                var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(grid);
                for (int i = 0; i < holders.Count; i++)
                {
                    var model = holders[i].CardModel;
                    bool isSelected = (selectedCards != null && selectedCards.Contains(model)) || (highlightedCards != null && highlightedCards.Contains(model));
                    cards.Add(new { 
                        index = i, 
                        name = model?.Title ?? "Unknown",
                        id = model?.Id.Entry ?? "unknown",
                        upgraded = model != null && (GetPropValue(model, "IsUpgraded", false) || GetPropValue(model, "TimesUpgraded", 0) > 0),
                        cost = model != null ? GetPropValue(model, "BaseCost", 0) : 0,
                        selected = isSelected
                    });
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
                
                // If there's a multi-preview variant, check it too (mirroring upgrade logic refinement)
                var fieldMulti = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_transformMultiPreviewContainer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var multi = fieldMulti?.GetValue(transformScreen) as CanvasItem;

                isConfirming = (preview != null && preview.Visible) || (multi != null && multi.Visible);
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
            else if (gridSelection is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen simpleScreen)
            {
                var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen).GetField("_confirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var confirmBtn = field?.GetValue(simpleScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                isConfirming = confirmBtn != null && confirmBtn.Visible && confirmBtn.IsEnabled;
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = gridSelection.GetType().Name,
                floor = runState.TotalFloor,
                seed = currentSeed,
                is_gym = _gymMode,
                cards = cards,
                is_confirming = isConfirming
            }, JsonOptions);
        }

        // 1.5. Check for global screens that aren't on overlay stack
        // Optimized: Instead of searching all nodes from GetTree().Root, find specific room node if it exists
        MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicCollection? relicCollection = null;
        var treasureRoomNodeForRelics = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom>(GetTree().Root).FirstOrDefault();
        if (treasureRoomNodeForRelics != null)
        {
            relicCollection = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicCollection>(treasureRoomNodeForRelics).FirstOrDefault(c => ((CanvasItem)c).Visible);
        }

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
        if (currentRoom == null) {
             Logger.Info("[AutoAI] currentRoom is null. Reporting game_over (transition or end).");
             return System.Text.Json.JsonSerializer.Serialize(new { type = "game_over", floor = runState.TotalFloor, seed = currentSeed, is_gym = _gymMode, victory = false, reason = "room_null" }, JsonOptions);
        }

        if (currentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom combatRoom)
        {
            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
            VerifyCombatPrediction(player, runState.TotalFloor);

            // Check if hand is in selection mode (e.g., Armaments, Grid selection in combat)
            // MUST do this before busy/queue check because selection is part of an action execution
            var handNode = MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Instance;
            if (handNode != null)
            {
                var confirmBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectModeConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var confirmBtn = confirmBtnField?.GetValue(handNode) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;

                bool isSelectionMode = handNode.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.SimpleSelect || 
                                     handNode.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.UpgradeSelect;

                if (isSelectionMode || handNode.IsInCardSelection)
                {
                    var selectedCardsField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectedCards", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var selectedCards = selectedCardsField?.GetValue(handNode) as IEnumerable<MegaCrit.Sts2.Core.Models.CardModel>;

                    var cards = new List<object>();
                    var activeHolders = handNode.ActiveHolders;
                    for (int i = 0; i < activeHolders.Count; i++)
                    {
                        var holder = activeHolders[i];
                        if (holder.CardNode != null)
                        {
                            var model = holder.CardNode.Model;
                            cards.Add(new { 
                                index = i, 
                                name = model?.Title ?? "Unknown",
                                selected = selectedCards != null && selectedCards.Contains(model)
                            });
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

            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            bool queueProcessing = rm.ActionQueueSet != null && !rm.ActionQueueSet.IsEmpty;

            bool combatBusy = cm != null && cm.IsInProgress && !cm.IsPlayPhase;
            if (combatBusy || queueProcessing) 
            {
                if (queueProcessing) {
                    Logger.Info("[AutoAI] combat_waiting: ActionQueue is not empty. Waiting for execution.");
                }
                return null;
            }

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

            // We no longer block here; we report the state and let Python decide (e.g. End Turn).
            // if (cm.PlayerActionsDisabled) return "{\"type\":\"combat_waiting\"}";

            var combatNode = MegaCrit.Sts2.Core.Nodes.Rooms.NCombatRoom.Instance;
            
            // Check for proceed button or dialogue hitboxes that block combat
            bool canProceed = combatNode?.ProceedButton?.IsEnabled ?? false;
            if (!canProceed)
            {
                // 1. Check for Ancient Dialogue Hitbox
                var ancientHitbox = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NAncientDialogueHitbox>(GetTree().Root);
                if (ancientHitbox != null && ancientHitbox.IsVisibleInTree() && ancientHitbox.IsEnabled)
                {
                    Logger.Info("[AutoAI] Detected active AncientDialogueHitbox in CombatRoom. Setting can_proceed=true.");
                    canProceed = true;
                }

                // 2. Check for Event Option Buttons (e.g. "FIGHT!" in a combat event)
                if (!canProceed)
                {
                    var eventOption = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>(GetTree().Root);
                    if (eventOption != null && eventOption.IsVisibleInTree() && eventOption.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Detected active EventOptionButton in CombatRoom. Setting can_proceed=true.");
                        canProceed = true;
                    }
                }

                // 3. Check for generic NDivinationButton (e.g. Wheel of Change)
                if (!canProceed)
                {
                    var divButton = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NDivinationButton>(GetTree().Root);
                    if (divButton != null && divButton.IsVisibleInTree() && divButton.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Detected active DivinationButton in CombatRoom. Setting can_proceed=true.");
                        canProceed = true;
                    }
                }
            }

            var combatData = new
            {
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
                    masterDeck = player?.Deck.Cards.Select(c => c.Id.Entry).ToList() ?? new List<string>(),
                    relics = player?.Relics.Select(r => r.Id.Entry).ToList() ?? new List<string>(),
                    powers = player?.Creature.Powers.Select(p => (object)new { id = p.Id.Entry, amount = p.Amount }).ToList() ?? new List<object>()
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
                        upgraded = GetPropValue(c, "IsUpgraded", false) || GetPropValue(c, "TimesUpgraded", 0) > 0,
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
                                repeats = repeats,
                                count = (intent is MegaCrit.Sts2.Core.MonsterMoves.Intents.StatusIntent si) ? si.CardCount : 0
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
                        isMinion = e.IsSecondaryEnemy,
                        powers = e.Powers.Select(p => (object)new { id = p.Id.Entry, amount = p.Amount }).ToList() ?? new List<object>(),
                        intents = intents
                    };
                }).ToList()
            };

            // --- Predicted Damage and Block ---
            int predictedTotalDamage = 0;
            foreach (var e in combatRoom.Enemies.Where(monster => monster.IsAlive))
            {
                if (e.Monster?.NextMove?.Intents != null)
                {
                    foreach (var intent in e.Monster.NextMove.Intents)
                    {
                        if (intent is MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent attackIntent)
                        {
                            predictedTotalDamage += attackIntent.GetSingleDamage(combatRoom.CombatState.PlayerCreatures, e) * attackIntent.Repeats;
                        }
                    }
                }
            }

        int predictedEndBlock = player?.Creature.Block ?? 0;
        if (player != null)
        {
            // Add Metallicize and Plated Armor
                foreach (var p in player.Creature.Powers)
                {
                    if (p.Id.Entry == "Metallicize" || p.Id.Entry == "Plated Armor")
                    {
                        predictedEndBlock += p.Amount;
                    }
                }

                // Relics: Orichalcum
                if (player.Relics.Any(r => r.Id.Entry == "Orichalcum"))
                {
                    // Orichalcum triggers if the player has 0 block at the end of the turn
                    if (predictedEndBlock == 0 && player.Creature.Block == 0)
                    {
                        predictedEndBlock += 6;
                    }
                }

                // Orbs (Frost)
                if (pState != null)
                {
                    try
                    {
                        var orbsProp = pState.GetType().GetProperty("Orbs");
                        var orbs = orbsProp?.GetValue(pState) as System.Collections.IEnumerable;
                        if (orbs != null)
                        {
                            foreach (var orb in orbs)
                            {
                                var idProp = orb.GetType().GetProperty("Id");
                                var idObj = idProp?.GetValue(orb);
                                string orbId = "";
                                if (idObj != null)
                                {
                                    var entryProp = idObj.GetType().GetProperty("Entry");
                                    orbId = entryProp?.GetValue(idObj) as string ?? "";
                                }
                                if (orbId == "Frost")
                                {
                                    var passiveProp = orb.GetType().GetProperty("PassiveAmount");
                                    predictedEndBlock += (int)(passiveProp?.GetValue(orb) ?? 0);
                                }
                            }
                        }
                    }
                    catch { }
                }
            }

            var resultDict = new Dictionary<string, object>
            {
                ["type"] = "combat",
                ["floor"] = runState.TotalFloor,
                ["seed"] = currentSeed,
                ["is_gym"] = _gymMode,
                ["can_proceed"] = canProceed,
                ["actions_disabled"] = cm.PlayerActionsDisabled,
                ["player"] = combatData.player,
                ["hand"] = combatData.hand,
                ["potions"] = combatData.potions,
                ["enemies"] = combatData.enemies,
                ["predicted_total_damage"] = predictedTotalDamage,
                ["predicted_end_block"] = predictedEndBlock,
                ["surplus_block"] = predictedEndBlock >= predictedTotalDamage,
                ["retains_block"] = player != null && player.Creature.Powers.Any(p => p.Id.Entry == "Barricade" || p.Id.Entry == "Blur")
            };

            var result = System.Text.Json.JsonSerializer.Serialize(resultDict, JsonOptions);

            // Update trackers (StepAI will set _waitingForVerification if needed)
            _lastPredictedDamage = predictedTotalDamage;
            _lastPredictedBlock = predictedEndBlock;
            _lastPlayerHp = player?.Creature.CurrentHp ?? 0;
            _lastFloor = runState.TotalFloor;
            _lastRetainsBlock = resultDict.ContainsKey("retains_block") && (bool)resultDict["retains_block"];

            return result;
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
            var eventRoomNode = MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Instance;

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
                var buttons = eventRoomNode != null ? FindNodesByType<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>(eventRoomNode)
                    .OrderBy(b => b.GlobalPosition.Y) // Usually top to bottom
                    .ToList() : new List<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>();

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

            // Check for proceed button
            var proceedBtn = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.CommonUi.NProceedButton>(eventRoomNode);
            bool canProceed = (proceedBtn != null && proceedBtn.IsEnabled && proceedBtn.IsVisibleInTree()) || ev.IsFinished;

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "event",
                floor = runState.TotalFloor,
                id = ev.Id.Entry,
                title = ev.Title.GetRawText(),
                options = options,
                can_proceed = canProceed
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
                var restSiteNodeForButtons = MegaCrit.Sts2.Core.Nodes.Rooms.NRestSiteRoom.Instance;
                var buttons = restSiteNodeForButtons != null ? FindNodesByType<MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton>(restSiteNodeForButtons)
                    .Where(b => b.Visible)
                    .OrderBy(b => b.GlobalPosition.X)
                    .ToList() : new List<MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton>();
                
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

            var merchantRoomNode = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
            var inventoryNode = merchantRoomNode != null ? FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory>(merchantRoomNode).FirstOrDefault(n => n.Visible) : null;
            if (inventoryNode != null && inventoryNode.Visible)
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
            var treasureRoomNodeForStatus = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom>(GetTree().Root).FirstOrDefault();
            bool hasChest = false;
            bool canProceed = false;

            if (treasureRoomNodeForStatus != null)
            {
                var chestBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom).GetField("_chestButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var chestBtn = chestBtnField?.GetValue(treasureRoomNodeForStatus) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                hasChest = chestBtn != null && chestBtn.Visible && chestBtn.IsEnabled;
                canProceed = treasureRoomNodeForStatus.ProceedButton?.IsEnabled ?? false;
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
        
        // Cache management: Check if floor or position changed or if we have no cache
        string currentBoss = "Unknown";
        try {
            var act = GetPropValue<object>(runState, "Act", null);
            if (act != null) {
                var bossEncounter = GetPropValue<object>(act, "BossEncounter", null);
                if (bossEncounter != null) {
                    var idObj = GetPropValue<object>(bossEncounter, "Id", null);
                    if (idObj != null) {
                        currentBoss = GetPropValue<string>(idObj, "Entry", currentBoss);
                    }
                }
            }
        } catch {}

        if (_lastMapJson != "" && _lastMapFloor == runState.TotalFloor && _lastMapBoss == currentBoss && 
            _lastMapPos?.row == currentPos?.row && _lastMapPos?.col == currentPos?.col)
        {
            return _lastMapJson;
        }

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
            var act = GetPropValue<object>(runState, "Act", null);
            if (act != null) {
                var bossEncounter = GetPropValue<object>(act, "BossEncounter", null);
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

        // Calculate next available nodes
        var nextNodes = new List<object>();
        if (!currentPos.HasValue)
        {
            // Search for the first non-empty row (up to row 5) for starting nodes
            for (int r = 0; r <= 5; r++)
            {
                var rowNodes = runState.Map.GetPointsInRow(r);
                if (rowNodes != null && rowNodes.Any())
                {
                    foreach (var p in rowNodes)
                    {
                        nextNodes.Add(new { row = p.coord.row, col = p.coord.col });
                    }
                    break;
                }
            }
        }
        else
        {
            var currentPoint = runState.Map.GetPoint(currentPos.Value);
            if (currentPoint != null && currentPoint.Children != null)
            {
                foreach (var child in currentPoint.Children)
                {
                    nextNodes.Add(new { row = child.coord.row, col = child.coord.col });
                }
            }
        }

        _lastMapJson = System.Text.Json.JsonSerializer.Serialize(new
        {
            type = "map",
            floor = runState.TotalFloor,
            seed = runState.Rng.StringSeed,
            is_gym = _gymMode,
            current_pos = currentPos.HasValue ? new { row = currentPos.Value.row, col = currentPos.Value.col } : null,
            next_nodes = nextNodes,
            nodes = nodes,
            edges = edges,
            boss = currentBoss
        }, JsonOptions);
        
        _lastMapFloor = runState.TotalFloor;
        _lastMapBoss = currentBoss;
        _lastMapPos = currentPos.HasValue ? (currentPos.Value.row, currentPos.Value.col) : null;

        return _lastMapJson;
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
