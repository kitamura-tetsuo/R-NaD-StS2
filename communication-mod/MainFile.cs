using Godot;

using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace communication_mod;

/// <summary>
/// Mod のエントリポイント。初期化とフック管理のみを担当します。
/// ポーリングなし – すべての AI トリガーはゲームイベントから発火します。
/// </summary>
[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string ModId = "communication-mod";

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    // --- AI trigger deduplication ---
    // Multiple hooks can fire in the same frame; this flag ensures only one
    // TriggerAI() execution is scheduled at a time.
    private bool _aiPending = false;
    private long _lastActionTime = 0;
    private int _lastTurnProcessed = -1;
    private bool _endTurnSentThisTurn = false;

    /// <summary>
    /// Called from every game-event hook. Schedules a deferred AI call.
    /// </summary>
    private void ScheduleAI()
    {
        if (AiBridge == null) return;
        if (_aiPending) 
        {
            // Logger.Info("[AutoAI] ScheduleAI: AI already pending.");
            return;
        }
        _aiPending = true;
        CallDeferred(nameof(TriggerAI));
    }

    /// <summary>
    /// Executes one AI step: reads game state → asks bridge → executes action.
    /// If the bridge returns "wait" this still clears _aiPending so the next hook
    /// can schedule a fresh attempt.
    /// </summary>
    private void TriggerAI()
    {
        _aiPending = false;

        if (AiBridge == null) return;

        // Ensure run-specific hooks (CombatManager, etc.) are registered for the current state.
        RegisterRunHooks();

        // Enforce fastest settings persistently
        Engine.TimeScale = 10.0f;
        try {
            var sm = MegaCrit.Sts2.Core.Saves.SaveManager.Instance;
            if (sm != null && sm.PrefsSave.FastMode != MegaCrit.Sts2.Core.Settings.FastModeType.Instant) {
                sm.PrefsSave.FastMode = MegaCrit.Sts2.Core.Settings.FastModeType.Instant;
                Logger.Info("[AutoAI] Persistent enforce: FastMode = Instant");
            }
        } catch (Exception) {}

        try
        {
            string stateJson = GetJsonState();
            var responseVariant = AiBridge.Call("predict_action", stateJson);
            if (responseVariant.VariantType == Variant.Type.Nil) return;

            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) != Error.Ok) return;

            var dict = json.Data.AsGodotDictionary();
            if (!dict.ContainsKey("action")) return;

            string action = dict["action"].AsString();

            // Rate limiting: ensure at least 500ms between actions
            long currentTime = System.DateTimeOffset.Now.ToUnixTimeMilliseconds();
            if (currentTime - _lastActionTime < 500 && action != "wait")
            {
                // Logger.Info($"[AutoAI] Rate limiting: skipping action={action} (too soon)");
                GetTree().CreateTimer(0.2).Connect("timeout", new Callable(this, nameof(ScheduleAI)));
                return;
            }
            _lastActionTime = currentTime;

            Logger.Info($"[AutoAI] TriggerAI: action={action}");

            if (action == "command")
            {
                string command = dict["command"].AsString();
                Logger.Info($"[AutoAI] Received command: {command}");
                ProcessCommand(command);
            }
            else if (action == "take_screenshot")
            {
                string path = dict["path"].AsString();
                Logger.Info($"[AutoAI] Received take_screenshot action for path: {path}");
                TakeScreenshot(path);
                AiBridge.Call("mark_screenshot_done");
                Logger.Info("[AutoAI] Screenshot done signaled to bridge. Re-scheduling AI.");
                ScheduleAI();
                return;
            }
            else if (action == "wait")
            {
                Logger.Info("[AutoAI] TriggerAI: received 'wait'. Scheduling retry in 1.0s.");
                // Always schedule a retry when the bridge says wait, or we might hang if
                // no more game events fire.
                GetTree().CreateTimer(1.0).Connect("timeout",
                    new Callable(this, nameof(ScheduleAI)));
            }
            else if (action == "end_turn")
            {
                var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
                var combatState = cm?.DebugOnlyGetState();
                int currentTurn = combatState?.RoundNumber ?? -1;

                if (currentTurn != _lastTurnProcessed)
                {
                    _lastTurnProcessed = currentTurn;
                    _endTurnSentThisTurn = false;
                }

                if (_endTurnSentThisTurn)
                {
                    Logger.Info("[AutoAI] Skipping duplicate end_turn for this turn.");
                    GetTree().CreateTimer(1.0).Connect("timeout", new Callable(this, nameof(ScheduleAI)));
                    return;
                }

                _endTurnSentThisTurn = true;
                ExecuteAction(dict);
            }
            else
            {
                ExecuteAction(dict);
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Error in TriggerAI: {ex.Message}");
        }
    }

    // -----------------------------------------------------------------------
    // Static wiring
    // -----------------------------------------------------------------------

    private static Node? _aiBridge;
    public static Node? AiBridge
    {
        get => _aiBridge;
        private set
        {
            _aiBridge = value;
            if (_aiBridge != null)
                RegisterGlobalHooks();
        }
    }

    private static MainFile? _instance;

    private static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new()
    {
        IncludeFields = true,
        PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
    };

    public static void Initialize()
    {
        Logger.Info("[AutoAI] MainFile.Initialize called");
        Engine.TimeScale = 100.0f;

        _instance = new MainFile();
        _instance.Name = "R_NaD_Controller";
        _instance?.CallDeferred(nameof(SafeSetup));
    }

    private static void SafeSetup()
    {
        try
        {
            SceneTree tree = (SceneTree)Engine.GetMainLoop();
            if (tree == null || tree.Root == null)
            {
                _instance?.GetTree()?.CreateTimer(1.0)
                    .Connect("timeout", new Callable(_instance, nameof(SafeSetup)));
                return;
            }

            if (_instance != null && _instance.GetParent() == null)
            {
                _instance.ProcessMode = ProcessModeEnum.Always;
                tree.Root.AddChild(_instance);
                _instance.SetProcess(true);
                Logger.Info("[AutoAI] MainFile added to SceneTree and processing enabled.");
            }

            GDExtensionManager.LoadExtension("res://ai_bridge.gdextension");

            AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
            if (AiBridge != null)
            {
                AiBridge.Name = "MyAiBridge";
                tree.Root.AddChild(AiBridge);
                Logger.Info("[AutoAI] Mod initialized and AiBridge attached.");
                // Fire an initial AI step so the bridge can issue start_game if needed.
                _instance?.CallDeferred(nameof(TriggerAI));

                // Start a watchdog timer that ensures AI periodically checks state
                // even if hooks are missed.
                _instance?.StartWatchdog();
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Initialization error: {ex.Message}");
        }
    }

    private int _processCount = 0;
    public override void _Process(double delta)
    {
        if (AiBridge == null) return;

        _processCount++;
        // Log every 600 frames (approx 10s at 60fps) to confirm it's ALIVE
        if (_processCount % 600 == 0) {
            Logger.Info("[AutoAI] _Process loop ALIVE");
        }

        try {
            // Synchronous screenshot check
            var reqVariant = AiBridge.Call("check_screenshot_request");
            if (reqVariant.VariantType != Variant.Type.Nil && reqVariant.AsString() != "")
            {
                string path = reqVariant.AsString();
                Logger.Info($"[AutoAI] Screenshot request detected for path: {path}");
                TakeScreenshot(path);
                AiBridge.Call("mark_screenshot_done");
                Logger.Info("[AutoAI] Screenshot done signaled to bridge.");
            }
        } catch (System.Exception ex) {
            if (_processCount % 600 == 0) {
                Logger.Error($"[AutoAI] Error in _Process polling: {ex.Message}");
            }
        }
    }

    private void StartWatchdog()
    {
        GetTree().CreateTimer(1.0).Connect("timeout", new Callable(this, nameof(OnWatchdogTimeout)));
    }

    private void OnWatchdogTimeout()
    {
        // Only trigger if no AI is currently pending.
        if (!_aiPending)
        {
            // Logger.Info("[AutoAI] Watchdog triggering ScheduleAI.");
            ScheduleAI();
        }
        StartWatchdog();
    }

    // -----------------------------------------------------------------------
    // Hook registration
    // -----------------------------------------------------------------------

    // Guards preventing double-registration on the same singleton instance.
    private static bool _overlayHooksRegistered = false;
    private static bool _mapHooksRegistered = false;
    private static bool _combatHooksRegistered = false;
    private static bool _actionExecutorHookRegistered = false;
    private static bool _roomEnteredHookRegistered = false;
    private static bool _runStartedHookRegistered = false;

    /// <summary>
    /// Registers hooks that are stable for the whole session (overlay stack, etc.).
    /// Called once when AiBridge becomes available.
    /// </summary>
    private static void RegisterGlobalHooks()
    {
        try
        {
            bool needsRetry = false;

            // Overlay opened/closed → rewards screen, card select, game over, etc.
            var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
            if (!_overlayHooksRegistered && overlayStack != null)
            {
                // Fire when a new overlay is pushed (rewards open, card select, game over, etc.).
                overlayStack.ChildEnteredTree += (_) => _instance?.ScheduleAI();
                _overlayHooksRegistered = true;
                Logger.Info("[AutoAI] Overlay hooks registered.");
            }
            else if (!_overlayHooksRegistered)
            {
                Logger.Info("[AutoAI] NOverlayStack.Instance is null – will retry hook registration.");
                needsRetry = true;
            }

            // Map (re-)opened → player can choose a node.
            var mapScreen = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance;
            if (!_mapHooksRegistered && mapScreen != null)
            {
                mapScreen.VisibilityChanged += () => {
                    if (mapScreen.Visible) _instance?.ScheduleAI();
                };
                _mapHooksRegistered = true;
                Logger.Info("[AutoAI] MapScreen visibility hook registered.");
            }
            else if (!_mapHooksRegistered)
            {
                needsRetry = true;
            }
            // Register per-run hooks whenever a new run starts.
            var rmInstance = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (rmInstance != null && !_runStartedHookRegistered) 
            {
                rmInstance.RunStarted += (_) => {
                    Logger.Info("[AutoAI] RunStarted event detected. Re-registering run hooks.");
                    _combatHooksRegistered = false;
                    _actionExecutorHookRegistered = false;
                    _roomEnteredHookRegistered = false;
                    RegisterRunHooks();
                };
                _runStartedHookRegistered = true;
            }

            // Register per-run hooks for current run if already in progress.
            RegisterRunHooks();

            // If some singletons weren't ready yet, retry after 2 seconds via instance method.
            if (needsRetry)
            {
                _instance?.GetTree()?.CreateTimer(2.0).Connect("timeout",
                    new Callable(_instance, nameof(RetryRegisterGlobalHooks)));
            }
        }
        catch (System.Exception ex)
        {
            Logger.Warn($"[AutoAI] RegisterGlobalHooks error: {ex.Message}");
        }
    }

    // Instance wrapper so Godot's timer Callable can invoke the static retry.
    private void RetryRegisterGlobalHooks() => RegisterGlobalHooks();

    /// <summary>
    /// Registers hooks that are tied to the current run's singleton objects.
    /// Must be called again each time a new run starts because CombatManager
    /// and RunManager.ActionExecutor are fresh instances per run.
    /// </summary>
    private static void RegisterRunHooks()
    {
        try
        {
            // Combat hooks
            var combatManager = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            if (!_combatHooksRegistered && combatManager != null)
            {
                // Player's turn begins
                combatManager.TurnStarted += (_) => _instance?.ScheduleAI();

                // Action queue freed (enemy animations done, etc.)
                combatManager.PlayerActionsDisabledChanged += (_) =>
                {
                    if (!combatManager.PlayerActionsDisabled)
                        _instance?.ScheduleAI();
                };

                _combatHooksRegistered = true;
                Logger.Info("[AutoAI] CombatManager hooks registered.");
            }

            // After each game-action finishes processing
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (rm != null)
            {
                if (!_actionExecutorHookRegistered && rm.ActionExecutor != null)
                {
                    rm.ActionExecutor.AfterActionExecuted += (_) => _instance?.ScheduleAI();
                    _actionExecutorHookRegistered = true;
                    Logger.Info("[AutoAI] RunManager.ActionExecutor hook registered.");
                }

                if (!_roomEnteredHookRegistered)
                {
                    // A new room has been entered (rest site, event, merchant, treasure, etc.).
                    rm.RoomEntered += () => {
                        _combatHooksRegistered = false;
                        _actionExecutorHookRegistered = false;
                        _instance?.ScheduleAI();
                    };
                    _roomEnteredHookRegistered = true;
                    Logger.Info("[AutoAI] RunManager.RoomEntered hook registered.");
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Warn($"[AutoAI] RegisterRunHooks error: {ex.Message}");
        }
    }

    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------

    private void ProcessCommand(string command)
    {
        Logger.Info($"[AutoAI] ProcessCommand: {command}");
        if (command.StartsWith("start_game"))
        {
            string seed = command.Contains(":") ? command.Split(':')[1] : "";
            Logger.Info($"[AutoAI] calling StartSts2Run with seed: '{seed}'");
            StartSts2Run(seed);
        }
        else if (command.StartsWith("screenshot:"))
        {
            TakeScreenshot(command.Substring("screenshot:".Length));
        }
    }

    private void StartSts2Run(string seed = "")
    {
        Logger.Info($"[AutoAI] StartSts2Run entered. seed: '{seed}'");
        try
        {
            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            Logger.Info($"[AutoAI] NGame.Instance check: {(ngame != null ? "FOUND" : "NULL")}");
            if (ngame != null)
            {
                _pendingSeed = seed;
                CallDeferred(nameof(StartSts2RunDeferred));
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] CRITICAL ERROR in StartSts2Run: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private string _pendingSeed = "";

    private async void StartSts2RunDeferred()
    {
        Logger.Info("[AutoAI] StartSts2RunDeferred entered.");
        try
        {
            string seedToUse = _pendingSeed;
            _pendingSeed = "";

            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            if (ngame == null)
            {
                Logger.Error("[AutoAI] NGame.Instance is NULL in deferred call!");
                return;
            }

            var ironclad = ModelDb.Character<Ironclad>();
            var acts     = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            Logger.Info($"[AutoAI] Calling StartNewSingleplayerRun with seed: {seedToUse}");
            await ngame.StartNewSingleplayerRun(ironclad, true, acts, modifiers, seedToUse, 0, null);

            Logger.Info("[AutoAI] StartNewSingleplayerRun returned. Waiting 2s for scene settle...");
            await Task.Delay(2000);

            // Reset per-run guards so the new run's CombatManager / ActionExecutor
            // get their hooks registered fresh.
            _combatHooksRegistered = false;
            _actionExecutorHookRegistered = false;
            _roomEnteredHookRegistered = false;
            RegisterRunHooks();

            var rm    = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            var state = rm.DebugOnlyGetState();

            if (state != null && state.CurrentRoom is MapRoom)
            {
                Logger.Info("[AutoAI] Entering first combat node...");
                await rm.EnterMapCoord(state.Map.StartingMapPoint.coord);
                Logger.Info("[AutoAI] Entered first combat node successfully.");
            }
            else
            {
                Logger.Warn($"[AutoAI] Not in MapRoom after start. Room: {state?.CurrentRoom?.GetType().Name ?? "null"}");
                // Trigger AI anyway so it can navigate whatever state it lands in.
                ScheduleAI();
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] CRITICAL ERROR in StartSts2RunDeferred: {ex.Message}\n{ex.StackTrace}");
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private void TakeScreenshot(string path)
    {
        Logger.Info($"[AutoAI] TakeScreenshot explicitly called for: {path}");
        try
        {
            var viewport = GetViewport();
            if (viewport == null) {
                Logger.Error("[AutoAI] GetViewport() returned null!");
                return;
            }
            var texture = viewport.GetTexture();
            if (texture == null) {
                Logger.Error("[AutoAI] viewport.GetTexture() returned null!");
                return;
            }
            var image = texture.GetImage();
            if (image != null)
            {
                string? dir = System.IO.Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(dir) && !System.IO.Directory.Exists(dir))
                    System.IO.Directory.CreateDirectory(dir);
                
                Error err = image.SavePng(path);
                if (err == Error.Ok) {
                    Logger.Info($"[AutoAI] Screenshot saved successfully to {path}");
                } else {
                    Logger.Error($"[AutoAI] image.SavePng failed with error: {err}");
                }
            }
            else
            {
                Logger.Error("[AutoAI] texture.GetImage() returned null!");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Screenshot exception: {ex.Message}\n{ex.StackTrace}");
        }
    }
}