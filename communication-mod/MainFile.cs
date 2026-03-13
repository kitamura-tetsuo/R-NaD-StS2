using Godot;

using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace communication_mod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string
        ModId = "communication-mod"; //At the moment, this is used only for the Logger and harmony names.

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    public static Node? AiBridge { get; private set; }
    private static MainFile? _instance;
    private static bool _diagnosed = false;

    private static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new()
    {
        IncludeFields = true,
        PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
    };

    public static void Initialize()
    {
        Logger.Info("[AutoAI] MainFile.Initialize called");
        Engine.TimeScale = 10.0f;

        _instance = new MainFile();
        _instance.Name = "R_NaD_Controller";
        
        SceneTree tree = (SceneTree)Engine.GetMainLoop();
        tree.Root.CallDeferred("add_child", _instance);

        try
        {
            // Manually load the GDExtension from the PCK
            string extensionPath = "res://ai_bridge.gdextension";
            Logger.Info($"Attempting to load GDExtension: {extensionPath}");
            GDExtensionManager.LoadExtension(extensionPath);

            AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
            AiBridge.Name = "MyAiBridge";
            tree.Root.CallDeferred("add_child", AiBridge);

            Logger.Info("Mod initialized and AiBridge attached.");
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Initialization error: {ex.Message}");
        }
    }

    private double _aiTimer = 0;
    private const double AiInterval = 2.0;

    public override void _Process(double delta)
    {
        _aiTimer += delta;
        if (_aiTimer >= AiInterval)
        {
            _aiTimer = 0;
            if (AiBridge != null)
            {
                PollCommands();
                
                // Allow UI to settle
                var transition = MegaCrit.Sts2.Core.Nodes.NGame.Instance?.Transition;
                if (transition != null && transition.InTransition) return;
                
                TryAiMove();
            }
        }
    }

    private void PollCommands()
    {
        if (AiBridge == null) return;

        try
        {
            // Send empty state to just poll for commands
            var responseVariant = AiBridge.Call("predict_action", "{}");
            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action") && dict["action"].AsString() == "command")
                {
                    string command = dict["command"].AsString();
                    Logger.Info($"[AutoAI] Received command: {command}");
                    if (command == "start_game")
                    {
                        StartNewGame();
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error polling commands: {ex.Message}");
        }
    }

    private void StartNewGame()
    {
        Logger.Info("[AutoAI] Starting new game...");
        try
        {
            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            Logger.Info($"[AutoAI] NGame.Instance: {(ngame != null ? "exists" : "null")}");
            if (ngame != null)
            {
                var ironclad = ModelDb.Character<Ironclad>();
                var acts = ActModel.GetDefaultList();
                var modifiers = new List<ModifierModel>();
                
                Logger.Info($"[AutoAI] Launching new run with {ironclad.Id.Entry}...");
                
                // Call the deferred method to ensure it runs on the main thread
                CallDeferred(nameof(StartNewGameDeferred));
            }
            else
            {
                Logger.Error("[AutoAI] NGame.Instance is null, cannot start game.");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error starting new game: {ex.Message}");
        }
    }

    private async void StartNewGameDeferred()
    {
        try
        {
            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            if (ngame == null) return;

            var ironclad = ModelDb.Character<Ironclad>();
            var acts = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            // StartNewSingleplayerRun handles the main run initialization
            var runState = await ngame.StartNewSingleplayerRun(
                ironclad, 
                true, 
                acts, 
                modifiers, 
                "", // Use random seed
                0, 
                null
            );
            
            Logger.Info("[AutoAI] Run started. Waiting for scene to settle...");
            await Task.Delay(2000); // Wait for transition effects
            
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            var state = rm.DebugOnlyGetState();
            
            if (state != null && state.CurrentRoom is MapRoom)
            {
                Logger.Info("[AutoAI] Landed in MapRoom. Entering starting combat node...");
                await rm.EnterMapCoord(state.Map.StartingMapPoint.coord);
                Logger.Info("[AutoAI] Successfully entered first node.");
            }
            else
            {
                string roomType = state?.CurrentRoom?.GetType().Name ?? "null";
                Logger.Info($"[AutoAI] Current room is {roomType}. No auto-map entry needed.");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Error in StartNewGameDeferred: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private void TryAiMove()
    {
        try
        {
            string stateJson = GetJsonState();
            if (stateJson == "{}" || stateJson == "{\"type\":\"none\"}") return;

            var responseVariant = AiBridge?.Call("predict_action", stateJson);
            if (responseVariant == null) return;
            
            string response = responseVariant.Value.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action"))
                {
                    ExecuteAction(dict);
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error in TryAiMove: {ex.Message}");
        }
    }

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
                    // Local player selection through synchronizer
                    var sync = MegaCrit.Sts2.Core.Runs.RunManager.Instance.RestSiteSynchronizer;
                    if (sync != null) {
                        await sync.ChooseLocalOption(index);
                    } else {
                        Logger.Error("[AutoAI] RestSiteSynchronizer is null");
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
            else if (action == "confirm_selection")
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
                }
                
                if (confirmBtn != null)
                {
                    Logger.Info($"[AutoAI] Confirming selection on {top.GetType().Name}");
                    confirmBtn.Call("ForceClick");
                }
                else
                {
                    Logger.Error($"[AutoAI] Could not find enabled Confirm button on {top?.GetType().Name}");
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
                Logger.Info("[AutoAI] Proceeding from rewards/terminal...");
                var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
                if (overlayStack?.Peek() is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen rewardsScreen)
                {
                    Logger.Info("[AutoAI] Found NRewardsScreen. Triggering proceed sequence.");
                    
                    // 1. Try to call the internal proceed method via reflection
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

                    // 2. Direct RM proceed as a secondary trigger
                    await rm.ProceedFromTerminalRewardsScreen();
                    
                    // 3. Force exit if RoomCount is 1 (the stall bug workaround)
                    if (runState.CurrentRoomCount == 1) {
                         Logger.Info("[AutoAI] RM count is 1, manually forcing room exit and travel enablement.");
                         var exitMethod = typeof(MegaCrit.Sts2.Core.Runs.RunManager).GetMethod("ExitCurrentRoom", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                         if (exitMethod != null) {
                             var task = exitMethod.Invoke(rm, null) as System.Threading.Tasks.Task;
                             if (task != null) await task;
                         }
                         MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance?.SetTravelEnabled(true);
                    }

                    // 4. Force remove the overlay to reveal the map
                    Logger.Info("[AutoAI] Force-removing NRewardsScreen from overlay stack.");
                    overlayStack.Remove(rewardsScreen);
                }
                else
                {
                    await rm.ProceedFromTerminalRewardsScreen();
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
             Godot.GD.Print($"[AutoAI] currentRoom is null. mapScreenExists={mapScreenExists}, mapScreenOpen={mapScreenOpen}");
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
                        rewards.Add(new { index = i, description = reward.Description.GetRawText() });
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

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "rewards",
                rewards = rewards,
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
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen gridScreen)
        {
            var cards = new List<object>();
            var gridField = typeof(MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen).GetField("_grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var grid = gridField?.GetValue(gridScreen) as MegaCrit.Sts2.Core.Nodes.Cards.NCardGrid;
            
            if (grid != null)
            {
                var holders = FindNodesByType<MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder>(grid);
                for (int i = 0; i < holders.Count; i++)
                {
                    cards.Add(new { index = i, name = holders[i].CardModel?.Title ?? "Unknown" });
                }
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = "grid_selection",
                cards = cards,
                can_skip = false // Typically grid screens don't have a simple skip
            }, JsonOptions);
        }
        else if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardGridSelectionScreen gridSelection)
        {
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

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "grid_selection",
                subtype = gridSelection.GetType().Name,
                cards = cards,
                is_confirming = isConfirming
            }, JsonOptions);
        }

        // 2. Room Logic
        if (currentRoom == null) return "{\"type\":\"none\"}";

        if (currentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom combatRoom)
        {
            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            if (cm == null || !cm.IsInProgress || !cm.IsPlayPhase || cm.PlayerActionsDisabled) return "{\"type\":\"combat_waiting\"}";

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
            for (int i = 0; i < options.Count; i++)
            {
                optionsData.Add(new
                {
                    index = i,
                    title = options[i].Title.GetRawText(),
                    is_enabled = options[i].IsEnabled
                });
            }

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "rest_site",
                options = optionsData
            }, JsonOptions);
        }


        else if (currentRoom is MegaCrit.Sts2.Core.Rooms.MerchantRoom)
        {
            // For the AI, always just proceed past the shop to avoid stalls
            var merchantRoom = MegaCrit.Sts2.Core.Nodes.Rooms.NMerchantRoom.Instance;
            bool canProceed = merchantRoom?.ProceedButton?.IsEnabled ?? false;
            return System.Text.Json.JsonSerializer.Serialize(new
            {
                type = "shop",
                can_proceed = canProceed
            }, JsonOptions);
        }


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
            IEnumerable<MegaCrit.Sts2.Core.Map.MapPoint> nextPoints;
            if (currentPos.HasValue)
            {
                var point = runState.Map.GetPoint(currentPos.Value);
                nextPoints = point?.Children ?? new HashSet<MegaCrit.Sts2.Core.Map.MapPoint>();
            }
            else
            {
                nextPoints = runState.Map.StartingMapPoint?.Children ?? new HashSet<MegaCrit.Sts2.Core.Map.MapPoint>();
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

    private List<T> FindNodesByType<T>(Node root) where T : class
    {
        var results = new List<T>();
        if (root is T t) results.Add(t);
        foreach (var child in root.GetChildren())
        {
            results.AddRange(FindNodesByType<T>(child));
        }
        return results;
    }
}