using Godot;
using System.Linq;
using System.Threading.Tasks;

namespace communication_mod;

public partial class MainFile : Node
{
    private async void ExecuteAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        Logger.Info($"[AutoAI] Executing action: {action}");

        try {
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            var runState = rm.DebugOnlyGetState();
            var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);

            if (action == "play_card")
            {
                string cardId = dict["card_id"].AsString();
                var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
                var combatState = cm.DebugOnlyGetState();
                var card = player.PlayerCombatState.Hand.Cards.FirstOrDefault(c => c.Id.Entry == cardId);

                if (card != null && card.CanPlay())
                {
                    // For simplicity, pick first valid target or none
                    MegaCrit.Sts2.Core.Entities.Creatures.Creature? target = null;
                    if (card.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                    {
                        target = combatState.Enemies.FirstOrDefault(e => e.IsAlive);
                    }

                    Logger.Info($"[AutoAI] Playing card: {card.Title}");
                    card.TryManualPlay(target);
                }
            }
            else if (action == "use_potion")
            {
                int index = (int)dict["index"].AsInt64();
                var potions = player.PotionSlots;
                if (index >= 0 && index < potions.Count)
                {
                    var potion = potions[index];
                    bool canUse = potion != null && potion.PassesCustomUsabilityCheck && (potion.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime || (potion.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly && MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress));
                    if (potion != null && canUse)
                    {
                        MegaCrit.Sts2.Core.Entities.Creatures.Creature? target = null;
                        if (potion.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                        {
                            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
                            var combatState = cm.DebugOnlyGetState();
                            target = combatState.Enemies.FirstOrDefault(e => e.IsAlive);
                        }
                        
                        Logger.Info($"[AutoAI] Using potion: {potion.Title.GetRawText()}");
                        potion.EnqueueManualUse(target);
                    }
                }
            }
            else if (action == "end_turn")
            {
                var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
                cm.SetReadyToEndTurn(player, true);
                Logger.Info("[AutoAI] End Turn requested.");
            }
            else if (action == "select_event_option")
            {
                int index = (int)dict["index"].AsInt64();
                var eventRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Instance;
                if (eventRoom != null)
                {
                    var eventRoomLayout = eventRoom.Layout;
                    if (eventRoomLayout != null)
                    {
                        // Get the option buttons from the layout's current options list
                        var er = runState?.CurrentRoom as MegaCrit.Sts2.Core.Rooms.EventRoom;
                        if (er != null)
                        {
                            var ev = er.LocalMutableEvent;
                            if (ev != null)
                            {
                                // If event is finished, we need to click the proceed option
                                if (ev.IsFinished)
                                {
                                    Logger.Info("[AutoAI] Event is finished, triggering proceed via NEventRoom.");
                                    await MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Proceed();
                                }
                                else
                                {
                                    var options = ev.CurrentOptions;
                                    if (index >= 0 && index < options.Count)
                                    {
                                        var option = options[index];
                                        Logger.Info($"[AutoAI] Selecting event option [{index}]: locked={option.IsLocked}");
                                        eventRoom.OptionButtonClicked(option, index);
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Fallback: try via synchronizer
                    Logger.Info($"[AutoAI] Fallback: selecting event option via synchronizer index={index}");
                    MegaCrit.Sts2.Core.Runs.RunManager.Instance.EventSynchronizer?.ChooseLocalOption(index);
                }
            }
            else if (action == "select_map_node")
            {
                int row = (int)dict["row"].AsInt64();
                int col = (int)dict["col"].AsInt64();
                // Fix: MapCoord takes (col, row)
                var coord = new MegaCrit.Sts2.Core.Map.MapCoord(col, row);
                Logger.Info($"[AutoAI] Entering map node: row={row}, col={col}");
                await roomManager_EnterMapCoord(coord);
            }
            else if (action == "select_reward")
            {
                int index = (int)dict["index"].AsInt64();
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var rewardsScreen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen;
                if (rewardsScreen != null)
                {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen).GetField("_rewardButtons", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var buttons = field?.GetValue(rewardsScreen) as System.Collections.IList;
                    if (buttons != null && index >= 0 && index < buttons.Count)
                    {
                        var btn = buttons[index];

                        // Fix: Check if button is enabled before clicking
                        if (btn is MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable && !clickable.IsEnabled)
                        {
                            Logger.Info($"[AutoAI] Reward button at index {index} is disabled. Skipping.");
                            return;
                        }

                        var rewardProp = btn.GetType().GetProperty("Reward");
                        var reward = rewardProp?.GetValue(btn) as MegaCrit.Sts2.Core.Rewards.Reward;
                        if (reward != null)
                        {
                            Logger.Info($"[AutoAI] Selecting reward at index {index}: {reward.Description.GetRawText()}");

                            // Use ForceClick to trigger the full reward claim logic (including Disabling the button)
                            if (btn is Godot.Node node)
                            {
                                node.Call("ForceClick");
                            }
                            else
                            {
                                // Fallback for cases where it's not a node (shouldn't happen for NRewardButton)
                                await reward.OnSelectWrapper();
                            }
                        }
                    }
                }
            }
            else if (action == "select_reward_card")
            {
                int index = (int)dict["index"].AsInt64();
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var screen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen;
                if (screen != null)
                {
                    var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(screen);
                    if (index >= 0 && index < holders.Count)
                    {
                        var holder = holders[index];
                        Logger.Info($"[AutoAI] Selecting card reward index {index}: {holder.CardModel?.Title}");
                        screen.Call("SelectCard", holder);
                    }
                }
            }
            else if (action == "click_reward_button")
            {
                int index = (int)dict["index"].AsInt64();
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var screen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen;
                if (screen != null)
                {
                    var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton>(screen);
                    if (index >= 0 && index < buttons.Count)
                    {
                        var btn = buttons[index];

                        // Fix: Check if button is enabled before clicking
                        if (btn is MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable && !clickable.IsEnabled)
                        {
                            Logger.Info($"[AutoAI] Reward alternative button at index {index} is disabled. Skipping.");
                            return;
                        }

                        Logger.Info($"[AutoAI] Clicking reward button index {index}: {btn.Name}");

                        // Try direct call first
                        btn.Call("ForceClick");

                        // Reflection fallback if it's a known button type
                        if (btn is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton)
                        {
                            Node current = btn;
                            MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen? ancestorScreen = null;
                            while (current != null)
                            {
                                if (current is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen s)
                                {
                                    ancestorScreen = s;
                                    break;
                                }
                                current = current.GetParent();
                            }

                            if (ancestorScreen != null)
                            {
                                var method = ancestorScreen.GetType().GetMethod("OnAlternateRewardSelected", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                                if (method != null)
                                {
                                    Logger.Info("[AutoAI] Fallback: Calling OnAlternateRewardSelected directly on " + ancestorScreen.Name);
                                    // Use integer value for enum (2 = DismissScreenAndRemoveReward)
                                    // Fix: Value 1 was KeepReward, 2 is RemoveReward.
                                    method.Invoke(ancestorScreen, new object[] { 2 });
                                }
                            }
                        }
                    }
                }
            }
            else if (action == "select_rest_site_option")
            {
                int index = (int)dict["index"].AsInt64();
                if (runState?.CurrentRoom is MegaCrit.Sts2.Core.Rooms.RestSiteRoom rsr)
                {
                    Logger.Info($"[AutoAI] Selecting rest site option index: {index}");
                    
                    // Prioritize UI button click as it handles the full lifecycle (synchronizer + UI updates)
                    var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton>(GetTree().Root)
                        .Where(b => b.Visible)
                        .OrderBy(b => b.GlobalPosition.X)
                        .ToList();
                    
                    if (index >= 0 && index < buttons.Count)
                    {
                        var btn = buttons[index];
                        Logger.Info($"[AutoAI] Clicking rest site button via UI: {btn.Option.Title.GetRawText()}");
                        btn.Call("ForceClick");
                        return;
                    }

                    // Fallback to synchronizer if UI button not found
                    Logger.Info($"[AutoAI] UI button not found. Falling back to synchronizer for rest site index {index}.");
                    var sync = MegaCrit.Sts2.Core.Runs.RunManager.Instance.RestSiteSynchronizer;
                    if (sync != null) {
                        await sync.ChooseLocalOption(index);
                    } else {
                        Logger.Error("[AutoAI] RestSiteSynchronizer is null and UI button not found");
                    }
                }
            }
            else if (action == "select_grid_card")
            {
                int index = (int)dict["index"].AsInt64();
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var top = overlayStack?.Peek();

                if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen chooseScreen)
                {
                    if (index == -1)
                    {
                        var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen).GetField("_skipButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var skipBtn = field?.GetValue(chooseScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                        if (skipBtn != null && skipBtn.IsEnabled)
                        {
                            Logger.Info("[AutoAI] Clicking skip button on NChooseACardSelectionScreen");
                            skipBtn.Call("ForceClick");
                        }
                    }
                    else
                    {
                        var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(chooseScreen);
                        if (index >= 0 && index < holders.Count)
                        {
                            Logger.Info($"[AutoAI] Selecting card at index {index} on NChooseACardSelectionScreen: {holders[index].CardModel?.Title}");
                            chooseScreen.Call("SelectHolder", holders[index]);
                        }
                    }
                }
                else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen gridScreen)
                {
                    var gridField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen).GetField("_grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var grid = gridField?.GetValue(gridScreen) as MegaCrit.Sts2.Core.Nodes.Cards.NCardGrid;

                    if (grid != null)
                    {
                         var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(grid);
                         if (index >= 0 && index < holders.Count)
                         {
                             Logger.Info($"[AutoAI] Selecting card at index {index} on NCardGridSelectionScreen: {holders[index].CardModel?.Title}");
                             // Use the internal handler for more reliable selection
                             grid.Call("OnHolderPressed", holders[index]);
                         }
                    }
                }
            }
            else if (action == "select_hand_card")
            {
                int index = (int)dict["index"].AsInt64();
                var hand = MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Instance;
                if (hand != null)
                {
                    bool isSelectionMode = hand.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.SimpleSelect || 
                                         hand.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.UpgradeSelect;

                    if (isSelectionMode || hand.IsInCardSelection)
                    {
                        var activeHolders = hand.ActiveHolders;
                        if (index >= 0 && index < activeHolders.Count)
                        {
                            var holder = activeHolders[index];
                            Logger.Info($"[AutoAI] Selecting hand card at index {index}: {holder.CardNode?.Model?.Title}");
                            hand.Call("OnHolderPressed", holder);
                            return;
                        }
                    }
                }

                // Fallback for NSimpleCardSelectScreen
                var simpleSelect = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen>(GetTree().Root).FirstOrDefault(s => s.Visible);
                if (simpleSelect != null)
                {
                    var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(simpleSelect);
                    if (index >= 0 && index < holders.Count)
                    {
                        var holder = holders[index];
                        Logger.Info($"[AutoAI] Selecting simple select card at index {index}: {holder.CardModel?.Title}");
                        simpleSelect.Call("SelectCard", holder);
                        return;
                    }
                }
            }
            else if (action == "confirm_selection")
            {
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var top = overlayStack?.Peek();

                Node? confirmBtn = null;
                // ... (existing grid selection confirm buttons) ...
                if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen upgradeScreen)
                {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_singlePreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    confirmBtn = field?.GetValue(upgradeScreen) as Node;
                    if (confirmBtn == null || !((Godot.CanvasItem)confirmBtn).Visible) {
                        field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        confirmBtn = field?.GetValue(upgradeScreen) as Node;
                    }
                }
                else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen transformScreen)
                {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_previewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    confirmBtn = field?.GetValue(transformScreen) as Node;
                }
                else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen cardSelectScreen)
                {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen).GetField("_previewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    confirmBtn = field?.GetValue(cardSelectScreen) as Node;
                }
                else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen enchantScreen)
                {
                    var singleField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_singlePreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var multiField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    
                    var singleBtn = singleField?.GetValue(enchantScreen) as Node;
                    var multiBtn = multiField?.GetValue(enchantScreen) as Node;
                    
                    if (singleBtn != null && ((CanvasItem)singleBtn).IsVisibleInTree()) confirmBtn = singleBtn;
                    else if (multiBtn != null && ((CanvasItem)multiBtn).IsVisibleInTree()) confirmBtn = multiBtn;
                }

                if (confirmBtn != null)
                {
                    Logger.Info($"[AutoAI] Confirming selection on {top.GetType().Name}");
                    confirmBtn.Call("ForceClick");
                }
                else
                {
                    // Check if hand confirm button is available
                    var hand = MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Instance;
                    if (hand != null)
                    {
                        var field = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectModeConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        confirmBtn = field?.GetValue(hand) as Node;
                        if (confirmBtn != null && ((CanvasItem)confirmBtn).IsVisibleInTree())
                        {
                            Logger.Info("[AutoAI] Confirming selection on NPlayerHand");
                            confirmBtn.Call("ForceClick");
                            return;
                        }
                    }

                    // Fallback for NSimpleCardSelectScreen
                    var simpleSelect = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen>(GetTree().Root).FirstOrDefault(s => s.Visible);
                    if (simpleSelect != null)
                    {
                        var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen).GetField("_confirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        confirmBtn = field?.GetValue(simpleSelect) as Node;
                        if (confirmBtn != null && ((CanvasItem)confirmBtn).IsVisibleInTree())
                        {
                            Logger.Info("[AutoAI] Confirming selection on NSimpleCardSelectScreen");
                            confirmBtn.Call("ForceClick");
                            return;
                        }
                    }

                    Logger.Error($"[AutoAI] Could not find enabled Confirm button on {top?.GetType().Name}, NPlayerHand, or NSimpleCardSelectScreen");
                }
            }
            else if (action == "buy_item")
            {
                int index = (int)dict["index"].AsInt64();
                var inventoryNode = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory>(GetTree().Root).FirstOrDefault(n => n.Visible);
                if (inventoryNode != null)
                {
                    var slots = inventoryNode.GetAllSlots().ToList();
                    if (index >= 0 && index < slots.Count)
                    {
                        var slot = slots[index];
                        if (slot.Entry != null && slot.Entry.IsStocked && slot.Entry.EnoughGold)
                        {
                            Logger.Info($"[AutoAI] Buying item at index {index}: {slot.Entry.GetType().Name}");
                            // Trigger purchase via UI click for consistency
                            slot.Call("ForceClick");
                        }
                    }
                }
            }
            else if (action == "shop_proceed")
            {
                var merchantRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
                if (merchantRoom != null)
                {
                    var proceedBtn = merchantRoom.ProceedButton;
                    if (proceedBtn != null && proceedBtn.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Clicking shop proceed button.");
                        proceedBtn.Call("ForceClick");
                    }
                    else
                    {
                        Logger.Info("[AutoAI] Shop proceed button not enabled yet. Waiting.");
                    }
                }
            }
            else if (action == "return_to_main_menu")
            {
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var gameOverScreen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen;
                if (gameOverScreen != null)
                {
                    Logger.Info("[AutoAI] Found GameOverScreen. Returning to main menu.");
                    // Check for mainMenuButton via reflection
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_mainMenuButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var mainMenuBtn = field?.GetValue(gameOverScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;

                    if (mainMenuBtn != null && mainMenuBtn.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Clicking Main Menu button.");
                        mainMenuBtn.Call("ForceClick");
                    }
                    else
                    {
                        // Fallback: try to find any button that might return us
                        var continueBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_continueButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var continueBtn = continueBtnField?.GetValue(gameOverScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                        if (continueBtn != null && continueBtn.IsEnabled)
                        {
                            Logger.Info("[AutoAI] Main menu button not ready, clicking Continue button first.");
                            continueBtn.Call("ForceClick");
                        }
                        else
                        {
                             Logger.Info("[AutoAI] No button on GameOverScreen is enabled yet. Waiting.");
                        }
                    }
                }
            }
            else if (action == "proceed")
            {
                Logger.Info("[AutoAI] Proceeding from room/screen...");
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                var currentRoom = runState.CurrentRoom;

                if (overlayStack?.Peek() is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen rewardsScreen)
                {
                    Logger.Info("[AutoAI] Found NRewardsScreen. Triggering proceed sequence.");
                    try {
                        var rsType = typeof(MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen);
                        var onProceed = rsType.GetMethod("OnProceedButtonPressed", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var proceedBtnField = rsType.GetField("_proceedButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var proceedBtn = proceedBtnField?.GetValue(rewardsScreen);

                        if (onProceed != null && proceedBtn != null) {
                            Logger.Info("[AutoAI] Invoking OnProceedButtonPressed via reflection.");
                            onProceed.Invoke(rewardsScreen, new object[] { proceedBtn });
                        }
                    } catch (System.Exception ex) {
                        Logger.Error($"[AutoAI] Error calling OnProceedButtonPressed: {ex.Message}");
                    }
                    await rm.ProceedFromTerminalRewardsScreen();
                    
                    if (runState.CurrentRoomCount == 1) {
                         Logger.Info("[AutoAI] RM count is 1, manually forcing room exit and travel enablement.");
                         var exitMethod = typeof(MegaCrit.Sts2.Core.Runs.RunManager).GetMethod("ExitCurrentRoom", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         if (exitMethod != null) {
                             var task = exitMethod.Invoke(rm, null) as System.Threading.Tasks.Task;
                             if (task != null) await task;
                         }
                         MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance?.SetTravelEnabled(true);
                    }
                    overlayStack.Remove(rewardsScreen);
                }
                else if (currentRoom is MegaCrit.Sts2.Core.Rooms.RestSiteRoom)
                {
                    var restSiteNode = MegaCrit.Sts2.Core.Nodes.Rooms.NRestSiteRoom.Instance;
                    if (restSiteNode?.ProceedButton != null && restSiteNode.ProceedButton.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Clicking Rest Site proceed button.");
                        restSiteNode.ProceedButton.Call("ForceClick");
                    }
                }
                else if (currentRoom is MegaCrit.Sts2.Core.Rooms.MerchantRoom)
                {
                    var merchantRoomNode = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
                    if (merchantRoomNode?.ProceedButton != null && merchantRoomNode.ProceedButton.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Clicking Merchant proceed button.");
                        merchantRoomNode.ProceedButton.Call("ForceClick");
                    }
                }
                else if (currentRoom is MegaCrit.Sts2.Core.Rooms.TreasureRoom)
                {
                    var treasureRoomNode = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom>(GetTree().Root).FirstOrDefault();
                    if (treasureRoomNode?.ProceedButton != null && treasureRoomNode.ProceedButton.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Clicking Treasure proceed button.");
                        treasureRoomNode.ProceedButton.Call("ForceClick");
                    }
                }
                else
                {
                    await rm.ProceedFromTerminalRewardsScreen();
                }
            }
            else if (action == "open_chest")
            {
                var treasureRoomNode = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom>(GetTree().Root).FirstOrDefault();
                if (treasureRoomNode != null)
                {
                    var field = typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NTreasureRoom).GetField("_chestButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var chestBtn = field?.GetValue(treasureRoomNode) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                    if (chestBtn != null && chestBtn.IsEnabled)
                    {
                        Logger.Info("[AutoAI] Opening treasure chest.");
                        chestBtn.Call("ForceClick");
                    }
                }
            }
            else if (action == "select_treasure_relic")
            {
                int index = (int)dict["index"].AsInt64();
                var relicCollection = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicCollection>(GetTree().Root).FirstOrDefault(c => ((CanvasItem)c).Visible);
                if (relicCollection != null)
                {
                    var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic.NTreasureRoomRelicHolder>(relicCollection);
                    var holder = holders.FirstOrDefault(h => h.Index == index);
                    if (holder != null && holder.IsEnabled)
                    {
                        Logger.Info($"[AutoAI] Selecting treasure relic index {index}: {holder.Relic?.Model?.Id.Entry}");
                        holder.Call("ForceClick");
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error executing action {action}: {ex.Message}");
        }
    }

    private async Task roomManager_EnterMapCoord(MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        await MegaCrit.Sts2.Core.Runs.RunManager.Instance.EnterMapCoord(coord);
    }
}
