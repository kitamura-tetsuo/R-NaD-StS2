using Godot;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;

namespace communication_mod;

public partial class MainFile : Node
{
    // --- Room/Screen Specialized Action Executors ---

    public async Task ExecuteUniversalAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "play_card": await HandlePlayCard(dict); break;
            case "use_potion": await HandleUsePotion(dict); break;
            case "end_turn": await HandleEndTurn(dict); break;
            case "select_event_option": await HandleSelectEventOption(dict); break;
            case "select_map_node": await HandleSelectMapNode(dict); break;
            case "select_reward": await HandleSelectReward(dict); break;
            case "select_reward_card": await HandleSelectRewardCard(dict); break;
            case "click_reward_button": await HandleClickRewardButton(dict); break;
            case "select_rest_site_option": await HandleSelectRestSiteOption(dict); break;
            case "select_grid_card": await HandleSelectGridCard(dict); break;
            case "select_hand_card": await HandleSelectHandCard(dict); break;
            case "confirm_selection": await HandleConfirmSelection(dict); break;
            case "buy_item": await HandleBuyItem(dict); break;
            case "shop_proceed": await HandleShopProceed(dict); break;
            case "return_to_main_menu": await HandleReturnToMainMenu(dict); break;
            case "proceed": await HandleProceed(dict); break;
            case "open_chest": await HandleOpenChest(dict); break;
            case "discard_potion": await HandleDiscardPotion(dict); break;
            case "select_treasure_relic": await HandleSelectTreasureRelic(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled universal action: {action}"); break;
        }
    }

    public async Task ExecuteCombatAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "play_card": await HandlePlayCard(dict); break;
            case "use_potion": await HandleUsePotion(dict); break;
            case "end_turn": await HandleEndTurn(dict); break;
            case "discard_potion": await HandleDiscardPotion(dict); break;
            case "select_hand_card": await HandleSelectHandCard(dict); break;
            case "confirm_selection": await HandleConfirmSelection(dict); break;
            case "select_grid_card": await HandleSelectGridCard(dict); break;
            case "proceed": await HandleProceed(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled combat action: {action}"); break;
        }
    }

    public async Task ExecuteEventAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "select_event_option": await HandleSelectEventOption(dict); break;
            case "select_grid_card": await HandleSelectGridCard(dict); break;
            case "confirm_selection": await HandleConfirmSelection(dict); break;
            case "proceed": await HandleProceed(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled event action: {action}"); break;
        }
    }

    public async Task ExecuteShopAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "buy_item": await HandleBuyItem(dict); break;
            case "shop_proceed": await HandleShopProceed(dict); break;
            case "select_grid_card": await HandleSelectGridCard(dict); break;
            case "confirm_selection": await HandleConfirmSelection(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled shop action: {action}"); break;
        }
    }

    public async Task ExecuteMapAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "select_map_node": await HandleSelectMapNode(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled map action: {action}"); break;
        }
    }

    public async Task ExecuteRestSiteAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "select_rest_site_option": await HandleSelectRestSiteOption(dict); break;
            case "select_grid_card": await HandleSelectGridCard(dict); break;
            case "confirm_selection": await HandleConfirmSelection(dict); break;
            case "proceed": await HandleProceed(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled rest site action: {action}"); break;
        }
    }

    public async Task ExecuteTreasureAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "open_chest": await HandleOpenChest(dict); break;
            case "select_treasure_relic": await HandleSelectTreasureRelic(dict); break;
            case "proceed": await HandleProceed(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled treasure action: {action}"); break;
        }
    }

    public async Task ExecuteRewardAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "select_reward": await HandleSelectReward(dict); break;
            case "select_reward_card": await HandleSelectRewardCard(dict); break;
            case "click_reward_button": await HandleClickRewardButton(dict); break;
            case "proceed": await HandleProceed(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled reward action: {action}"); break;
        }
    }

    public async Task ExecuteGameOverAction(Godot.Collections.Dictionary dict)
    {
        string action = dict["action"].AsString();
        switch (action)
        {
            case "return_to_main_menu": await HandleReturnToMainMenu(dict); break;
            default: Logger.Warn($"[AutoAI] Unhandled game over action: {action}"); break;
        }
    }


    // --- Atomic Action Handlers ---

    public async Task HandlePlayCard(Godot.Collections.Dictionary dict)
    {
        string cardId = dict["card_id"].AsString();
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        var combatState = cm.DebugOnlyGetState();
        int cardIdx = dict.ContainsKey("card_index") ? (int)dict["card_index"].AsInt64() : -1;
        var hand = player.PlayerCombatState.Hand.Cards;
        var card = (cardIdx >= 0 && cardIdx < hand.Count) ? hand[cardIdx] : hand.FirstOrDefault(c => c.Id.Entry == cardId);

        if (card != null)
        {
            bool canPlay = card.CanPlay();
            var tt = card.TargetType;
            Logger.Info($"[AutoAI] Playing card: {card.Title} (ID: {cardId}, CanPlay: {canPlay}, TargetType: {tt})");

            if (canPlay)
            {
                MegaCrit.Sts2.Core.Entities.Creatures.Creature? target = null;
                
                // Only resolve target for types that actually require a selection (AnyEnemy, AnyAlly, AnyPlayer)
                // RandomEnemy and AllEnemies should be played with null target.
                if (tt == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                {
                    int targetIdx = dict.ContainsKey("target_index") ? (int)dict["target_index"].AsInt64() : 0;
                    var aliveEnemies = combatState.Enemies.Where(e => e.IsAlive).ToList();
                    if (targetIdx >= 0 && targetIdx < aliveEnemies.Count)
                    {
                        target = aliveEnemies[targetIdx];
                    }
                    else
                    {
                        target = aliveEnemies.FirstOrDefault();
                    }
                    Logger.Info($"[AutoAI] Resolved target for {tt}: {target?.Name ?? "None"} (Index: {targetIdx})");
                }
                else if (tt == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyAlly || tt == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyPlayer)
                {
                    target = player.Creature;
                    Logger.Info($"[AutoAI] Resolved target for {tt}: {target?.Name ?? "None"}");
                }

                bool result = card.TryManualPlay(target);
                Logger.Info($"[AutoAI] TryManualPlay result for {card.Title}: {result} (Target: {target?.Name ?? "None"})");
            }
            else
            {
                Logger.Info($"[AutoAI] Cannot play card: {card.Title} (CanPlay returned false)");
            }
        }
        else
        {
            Logger.Info($"[AutoAI] Card not found in hand: {cardId}");
        }
    }

    public async Task HandleUsePotion(Godot.Collections.Dictionary dict)
    {
        int index = (int)dict["index"].AsInt64();
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        var potions = player.PotionSlots;
        if (index >= 0 && index < potions.Count)
        {
            var potion = potions[index];
            bool canUse = potion != null && potion.PassesCustomUsabilityCheck && (potion.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime || (potion.Usage == MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly && MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress));
            if (potion != null && canUse)
            {
                MegaCrit.Sts2.Core.Entities.Creatures.Creature? target = null;
                string targetType = potion.TargetType.ToString();
                
                if (targetType.Contains("Enemy") || targetType.Contains("Single"))
                {
                    int targetIdx = dict.ContainsKey("target_index") ? (int)dict["target_index"].AsInt64() : 0;
                    var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
                    var combatState = cm.DebugOnlyGetState();
                    var aliveEnemies = combatState.Enemies.Where(e => e.IsAlive).ToList();
                    if (targetIdx >= 0 && targetIdx < aliveEnemies.Count)
                    {
                        target = aliveEnemies[targetIdx];
                    }
                    else
                    {
                        target = aliveEnemies.FirstOrDefault();
                    }
                    Logger.Info($"[AutoAI] Potion targeted enemy index {targetIdx}: {target?.Name ?? "None"}");
                }
                else if (targetType.Contains("Player") || targetType.Contains("Ally") || targetType == "Self")
                {
                    target = player.Creature;
                    Logger.Info($"[AutoAI] Potion targeted player: {target?.Name ?? "None"}");
                }
                
                // Fallback for single-target potions that failed to resolve a target
                if (target == null && potion.TargetType.IsSingleTarget())
                {
                    target = player.Creature;
                    Logger.Warn($"[AutoAI] Potion {potion.Title.GetRawText()} is IsSingleTarget but target was null. Falling back to player.");
                }
                
                Logger.Info($"[AutoAI] Using potion: {potion.Title.GetRawText()} (TargetType: {targetType}, ResolvedTarget: {target?.Name ?? "None"})");
                potion.EnqueueManualUse(target);
            }
            else
            {
                Logger.Info($"[AutoAI] Cannot use potion at index {index}: (potion exists: {potion != null}, canUse: {canUse})");
            }
        }
    }

    public async Task HandleEndTurn(Godot.Collections.Dictionary dict)
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        var combatState = cm.DebugOnlyGetState();
        int currentRound = combatState?.RoundNumber ?? 0;
        
        Logger.Info($"[AutoAI] Executing End Turn (Player: {player.Creature.Name}, Round: {currentRound})");
        
        var endTurnAction = new MegaCrit.Sts2.Core.GameActions.EndPlayerTurnAction(player, currentRound);
        rm.ActionQueueSynchronizer.RequestEnqueue(endTurnAction);
        
        Logger.Info("[AutoAI] EndPlayerTurnAction requested via synchronizer.");
    }

    public async Task HandleSelectEventOption(Godot.Collections.Dictionary dict)
    {
        int index = (int)dict["index"].AsInt64();
        var eventRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Instance;
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        
        if (eventRoom != null)
        {
            var er = runState?.CurrentRoom as MegaCrit.Sts2.Core.Rooms.EventRoom;
            var ev = er?.LocalMutableEvent;

            if (ev != null)
            {
                if (ev.IsFinished)
                {
                    Logger.Info("[AutoAI] Event is finished, triggering proceed via NEventRoom.");
                    await MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Proceed();
                    return;
                }

                var options = ev.CurrentOptions;
                if (index >= 0 && index < options.Count)
                {
                    var option = options[index];
                    Logger.Info($"[AutoAI] Selecting event option [{index}]: {option.Title.GetFormattedText()}");
                    eventRoom.OptionButtonClicked(option, index);
                    return;
                }
            }

            var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>(eventRoom)
                .OrderBy(b => b.GlobalPosition.Y)
                .ToList();

            if (index >= 0 && index < buttons.Count)
            {
                var btn = buttons[index];
                Logger.Info($"[AutoAI] Selecting event option via UI button index {index} (Button: {btn.Name})");
                if (btn.Option != null) {
                    eventRoom.OptionButtonClicked(btn.Option, index);
                } else {
                    btn.Call("OnReleased"); 
                }
                return;
            }
        }
        
        Logger.Error($"[AutoAI] Could not find event option index {index} on {eventRoom?.GetType().Name ?? "null"}");
    }

    public async Task HandleSelectMapNode(Godot.Collections.Dictionary dict)
    {
        int row = (int)dict["row"].AsInt64();
        int col = (int)dict["col"].AsInt64();
        var coord = new MegaCrit.Sts2.Core.Map.MapCoord(col, row);
        Logger.Info($"[AutoAI] Entering map node: row={row}, col={col}");
        await roomManager_EnterMapCoord(coord);
    }

    public async Task HandleSelectReward(Godot.Collections.Dictionary dict)
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
                    if (btn is Godot.Node node)
                    {
                        node.Call("ForceClick");
                    }
                    else
                    {
                        await reward.OnSelectWrapper();
                    }
                }
            }
        }
    }

    public async Task HandleSelectRewardCard(Godot.Collections.Dictionary dict)
    {
        int index = (int)dict["index"].AsInt64();
        var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
        var screen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardSelectionScreen;
        if (screen != null)
        {
            if (index == -1)
            {
                var buttons = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton>(screen);
                if (buttons.Count > 0)
                {
                    var btn = buttons[0];
                    Logger.Info($"[AutoAI] HandleSelectRewardCard: index=-1 detected. Clicking alternative button index 0: {btn.Name}");
                    btn.Call("ForceClick");

                    var method = screen.GetType().GetMethod("OnAlternateRewardSelected", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (method != null)
                    {
                        Logger.Info("[AutoAI] HandleSelectRewardCard fallback: Calling OnAlternateRewardSelected directly.");
                        method.Invoke(screen, new object[] { 2 });
                    }
                }
                else
                {
                    Logger.Warn("[AutoAI] HandleSelectRewardCard: index=-1 detected but no alternative buttons found.");
                }
            }
            else
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
    }

    public async Task HandleClickRewardButton(Godot.Collections.Dictionary dict)
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

                if (btn is MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable && !clickable.IsEnabled)
                {
                    Logger.Info($"[AutoAI] Reward alternative button at index {index} is disabled. Skipping.");
                    return;
                }

                Logger.Info($"[AutoAI] Clicking reward button index {index}: {btn.Name}");
                btn.Call("ForceClick");

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
                            method.Invoke(ancestorScreen, new object[] { 2 });
                        }
                    }
                }
            }
        }
    }

    public async Task HandleSelectRestSiteOption(Godot.Collections.Dictionary dict)
    {
        int index = (int)dict["index"].AsInt64();
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        if (runState?.CurrentRoom is MegaCrit.Sts2.Core.Rooms.RestSiteRoom rsr)
        {
            Logger.Info($"[AutoAI] Selecting rest site option index: {index}");
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

            Logger.Info($"[AutoAI] UI button not found. Falling back to synchronizer for rest site index {index}.");
            var sync = MegaCrit.Sts2.Core.Runs.RunManager.Instance.RestSiteSynchronizer;
            if (sync != null) {
                await sync.ChooseLocalOption(index);
            } else {
                Logger.Error("[AutoAI] RestSiteSynchronizer is null and UI button not found");
            }
        }
    }

    public async Task HandleSelectGridCard(Godot.Collections.Dictionary dict)
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
            var type = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen);
            var gridField = type.GetField("_grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (gridField == null) {
                gridField = top.GetType().GetField("_grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            }
            var grid = gridField?.GetValue(gridScreen) as MegaCrit.Sts2.Core.Nodes.Cards.NCardGrid;
            if (grid == null) {
                Logger.Warn($"[AutoAI] Could NOT find _grid field on {top.GetType().Name}. gridField exists? {gridField != null}");
            }

            if (grid != null)
            {
                 var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(grid);
                 if (index >= 0 && index < holders.Count)
                 {
                     Logger.Info($"[AutoAI] Selecting card at index {index} on {top.GetType().Name}: {holders[index].CardModel?.Title}");
                     grid.Call("OnHolderPressed", holders[index]);

                     if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen upgradeScreenAuto)
                     {
                         Logger.Info("[AutoAI] Automatically confirming upgrade selection.");
                         await Task.Delay(500); 

                         var singleFieldU = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_singlePreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         var multiFieldU = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckUpgradeSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

                         var singleBtnU = singleFieldU?.GetValue(upgradeScreenAuto) as Node;
                         var multiBtnU = multiFieldU?.GetValue(upgradeScreenAuto) as Node;

                         Node? confirmBtnU = null;
                         if (singleBtnU != null && ((CanvasItem)singleBtnU).IsVisibleInTree()) confirmBtnU = singleBtnU;
                         else if (multiBtnU != null && ((CanvasItem)multiBtnU).IsVisibleInTree()) confirmBtnU = multiBtnU;

                         if (confirmBtnU != null)
                         {
                             Logger.Info("[AutoAI] Clicking upgrade confirm button automatically.");
                             confirmBtnU.Call("ForceClick");
                         }
                     }
                     if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen enchantScreen)
                     {
                         Logger.Info("[AutoAI] Automatically confirming enchantment selection.");
                         await Task.Delay(500);
                         
                         var singleField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_singlePreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         var multiField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckEnchantSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         
                         var singleBtn = singleField?.GetValue(enchantScreen) as Node;
                         var multiBtn = multiField?.GetValue(enchantScreen) as Node;
                         
                         Node? confirmBtn = null;
                         if (singleBtn != null && ((CanvasItem)singleBtn).IsVisibleInTree()) confirmBtn = singleBtn;
                         else if (multiBtn != null && ((CanvasItem)multiBtn).IsVisibleInTree()) confirmBtn = multiBtn;

                         if (confirmBtn != null)
                         {
                             Logger.Info("[AutoAI] Clicking confirm button automatically.");
                             confirmBtn.Call("ForceClick");
                         }
                     }
                     if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen deckCardSelectScreen)
                     {
                         Logger.Info($"[AutoAI] Automatically confirming deck card selection on {deckCardSelectScreen.GetType().Name}");
                         await Task.Delay(500); 

                         var field = deckCardSelectScreen.GetType().GetField("_previewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         var confirmBtn = field?.GetValue(deckCardSelectScreen) as Node;
                         
                         if (confirmBtn == null || !((CanvasItem)confirmBtn).IsVisibleInTree())
                         {
                             field = deckCardSelectScreen.GetType().GetField("_confirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                             confirmBtn = field?.GetValue(deckCardSelectScreen) as Node;
                         }

                         if (confirmBtn != null && ((CanvasItem)confirmBtn).IsVisibleInTree())
                         {
                             Logger.Info("[AutoAI] Clicking deck card confirm button automatically.");
                             confirmBtn.Call("ForceClick");
                         }
                     }
                     if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen transformSelectScreen)
                     {
                         Logger.Info("[AutoAI] Automatically confirming transform selection.");
                         await Task.Delay(500); 

                         var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_previewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         var multiField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         
                         var singleBtn = field?.GetValue(transformSelectScreen) as Node;
                         var multiBtn = multiField?.GetValue(transformSelectScreen) as Node;
                         
                         Node? confirmBtn = null;
                         if (singleBtn != null && ((CanvasItem)singleBtn).IsVisibleInTree()) confirmBtn = singleBtn;
                         else if (multiBtn != null && ((CanvasItem)multiBtn).IsVisibleInTree()) confirmBtn = multiBtn;

                         if (confirmBtn != null)
                         {
                             Logger.Info("[AutoAI] Clicking transform confirm button automatically.");
                             confirmBtn.Call("ForceClick");
                         }
                     }
                 }
            }
        }
    }

    public async Task HandleSelectHandCard(Godot.Collections.Dictionary dict)
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

                    if (hand.CurrentMode == MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand.Mode.UpgradeSelect)
                    {
                        Logger.Info("[AutoAI] UpgradeSelect mode: waiting for confirm button to become available.");
                        await Task.Delay(500);
                        var confirmBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Combat.NPlayerHand).GetField("_selectModeConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        var confirmBtnNode = confirmBtnField?.GetValue(hand) as Node;
                        if (confirmBtnNode != null && ((CanvasItem)confirmBtnNode).IsVisibleInTree())
                        {
                            Logger.Info("[AutoAI] Clicking UpgradeSelect confirm button automatically.");
                            confirmBtnNode.Call("ForceClick");
                        }
                    }
                    return;
                }
            }
        }

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

    public async Task HandleConfirmSelection(Godot.Collections.Dictionary dict)
    {
        var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
        var top = overlayStack?.Peek();

        Node? confirmBtn = null;
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
            if (confirmBtn == null || !((Godot.CanvasItem)confirmBtn).IsVisibleInTree()) {
                var fieldMulti = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckTransformSelectScreen).GetField("_multiPreviewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                confirmBtn = fieldMulti?.GetValue(transformScreen) as Node;
            }
        }
        else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NDeckCardSelectScreen cardSelectScreen)
        {
            var field = cardSelectScreen.GetType().GetField("_previewConfirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            confirmBtn = field?.GetValue(cardSelectScreen) as Node;
            if (confirmBtn == null || !((Godot.CanvasItem)confirmBtn).IsVisibleInTree())
            {
                field = cardSelectScreen.GetType().GetField("_confirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                confirmBtn = field?.GetValue(cardSelectScreen) as Node;
            }
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
        else if (top is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen simpleScreen)
        {
            var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NSimpleCardSelectScreen).GetField("_confirmButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            confirmBtn = field?.GetValue(simpleScreen) as Node;
        }

        if (confirmBtn != null)
        {
            Logger.Info($"[AutoAI] Confirming selection on {top.GetType().Name}");
            confirmBtn.Call("ForceClick");
        }
        else
        {
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

    public async Task HandleBuyItem(Godot.Collections.Dictionary dict)
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
                    await slot.Entry.OnTryPurchaseWrapper(inventoryNode.Inventory);
                }
            }
        }
    }

    public async Task HandleShopProceed(Godot.Collections.Dictionary dict)
    {
        var merchantRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
        if (merchantRoom != null)
        {
            if (merchantRoom.Inventory != null && merchantRoom.Inventory.Visible)
            {
                var backButton = FindNodesByType<MegaCrit.Sts2.Core.Nodes.CommonUi.NBackButton>(merchantRoom.Inventory).FirstOrDefault();
                if (backButton != null && backButton.IsEnabled)
                {
                    Logger.Info("[AutoAI] Closing shop inventory via BackButton before proceeding.");
                    backButton.Call("ForceClick");
                    await Task.Delay(500);
                }
            }

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

    public async Task HandleReturnToMainMenu(Godot.Collections.Dictionary dict)
    {
        var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
        var gameOverScreen = overlayStack?.Peek() as MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen;
        
        bool clicked = false;
        if (gameOverScreen != null)
        {
            Logger.Info("[AutoAI] Found GameOverScreen. Returning to main menu.");
            var field = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_mainMenuButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var mainMenuBtn = field?.GetValue(gameOverScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;

            if (mainMenuBtn != null && mainMenuBtn.IsEnabled)
            {
                Logger.Info("[AutoAI] Clicking Main Menu button.");
                mainMenuBtn.Call("ForceClick");
                clicked = true;
            }
            else
            {
                var continueBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_continueButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var continueBtn = continueBtnField?.GetValue(gameOverScreen) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                if (continueBtn != null && continueBtn.IsEnabled)
                {
                    Logger.Info("[AutoAI] Main menu button not ready, clicking Continue button first.");
                    continueBtn.Call("ForceClick");
                    clicked = true;
                }
            }
        }

        if (!clicked)
        {
            Logger.Info("[AutoAI] No enabled button found on GameOverScreen (or screen not present). Using NGame.Instance.ReturnToMainMenu() forced fallback.");
            var nGame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            if (nGame != null)
            {
                await nGame.ReturnToMainMenu();
            }
            else
            {
                Logger.Error("[AutoAI] Could not perform ReturnToMainMenu: NGame.Instance is null");
            }
        }
    }

    public async Task HandleProceed(Godot.Collections.Dictionary dict)
    {
        Logger.Info("[AutoAI] Proceeding from room/screen...");
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
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
        else if (overlayStack?.Peek() is MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen gos)
        {
            Logger.Info("[AutoAI] Found GameOverScreen in HandleProceed. Clicking Continue button.");
            var continueBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_continueButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var continueBtn = continueBtnField?.GetValue(gos) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
            if (continueBtn != null && continueBtn.IsEnabled)
            {
                continueBtn.Call("ForceClick");
            }
            else
            {
                var mainMenuBtnField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen).GetField("_mainMenuButton", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var mainMenuBtn = mainMenuBtnField?.GetValue(gos) as MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton;
                if (mainMenuBtn != null && mainMenuBtn.IsEnabled)
                {
                    Logger.Info("[AutoAI] Continue button not enabled, but Main Menu button is. Clicking it.");
                    mainMenuBtn.Call("ForceClick");
                }
            }
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
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom)
        {
            var combatRoomNode = MegaCrit.Sts2.Core.Nodes.Rooms.NCombatRoom.Instance;
            if (combatRoomNode?.ProceedButton != null && combatRoomNode.ProceedButton.IsEnabled)
            {
                Logger.Info("[AutoAI] Clicking Combat proceed button (Victory Bag).");
                combatRoomNode.ProceedButton.Call("ForceClick");
            }
            else
            {
                // 1. Check for Ancient Dialogue Hitbox
                var ancientHitbox = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NAncientDialogueHitbox>(GetTree().Root);
                if (ancientHitbox != null && ancientHitbox.IsVisibleInTree() && ancientHitbox.IsEnabled)
                {
                    Logger.Info("[AutoAI] Clicking AncientDialogueHitbox in CombatRoom.");
                    ancientHitbox.Call("ForceClick");
                    return;
                }

                // 2. Check for Event Option Buttons
                var eventOption = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton>(GetTree().Root);
                if (eventOption != null && eventOption.IsVisibleInTree() && eventOption.IsEnabled)
                {
                    Logger.Info("[AutoAI] Clicking EventOptionButton in CombatRoom.");
                    eventOption.Call("ForceClick");
                    return;
                }

                // 3. Check for Divination Button
                var divButton = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NDivinationButton>(GetTree().Root);
                if (divButton != null && divButton.IsVisibleInTree() && divButton.IsEnabled)
                {
                    Logger.Info("[AutoAI] Clicking DivinationButton in CombatRoom.");
                    divButton.Call("ForceClick");
                    return;
                }
            }
        }
        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom)
        {
            Logger.Info("[AutoAI] Found EventRoom in HandleProceed. Triggering NEventRoom.Proceed().");
            await MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Proceed();
        }
        else
        {
            await rm.ProceedFromTerminalRewardsScreen();
        }
    }

    public async Task HandleOpenChest(Godot.Collections.Dictionary dict)
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

    public async Task HandleDiscardPotion(Godot.Collections.Dictionary dict)
    {
        int index = (int)dict["index"].AsInt64();
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
        var runState = rm.DebugOnlyGetState();
        var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        var potions = player.PotionSlots;
        if (index >= 0 && index < potions.Count)
        {
            var potion = potions[index];
            if (potion != null)
            {
                Logger.Info($"[AutoAI] Discarding potion: {potion.Title.GetRawText()} (Index: {index})");
                try {
                    var method = potion.GetType().GetMethod("Discard", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                    if (method != null) {
                        method.Invoke(potion, null);
                    } else {
                        Logger.Error($"[AutoAI] Could not find Discard method on {potion.GetType().FullName}");
                    }
                } catch (Exception ex) {
                    Logger.Error($"[AutoAI] Error discarding potion: {ex.Message}");
                }
            }
        }
    }

    public async Task HandleSelectTreasureRelic(Godot.Collections.Dictionary dict)
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

    private async Task roomManager_EnterMapCoord(MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        await MegaCrit.Sts2.Core.Runs.RunManager.Instance.EnterMapCoord(coord);
    }
}
