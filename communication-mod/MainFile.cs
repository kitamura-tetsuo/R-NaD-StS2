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

        // Enforce fastest settings persistently
        // Engine.TimeScale = 10.0f;
        // try {
        //     var sm = MegaCrit.Sts2.Core.Saves.SaveManager.Instance;
        //     if (sm != null && sm.PrefsSave.FastMode != MegaCrit.Sts2.Core.Settings.FastModeType.Instant) {
        //         sm.PrefsSave.FastMode = MegaCrit.Sts2.Core.Settings.FastModeType.Instant;
        //         Logger.Info("[AutoAI] Persistent enforce: FastMode = Instant");
        //     }
        // } catch (Exception) {}

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

            // Rate limiting: ensure at least 100ms between actions (relaxed from 500ms for high TimeScale)
            long currentTime = System.DateTimeOffset.Now.ToUnixTimeMilliseconds();
            if (currentTime - _lastActionTime < 100 && action != "wait")
            {
                // Logger.Info($"[AutoAI] Rate limiting: skipping action={action} (too soon)");
                GetTree().CreateTimer(0.05).Connect("timeout", new Callable(this, nameof(ScheduleAI)));
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
                // No longer need to schedule a timer here; 
                // _Process polling will catch the "wait" state and try again.
                return;
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
                    // No longer need timer; _Process will retry
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

            // High-frequency polling safety net: 
            // Trigger AI every frame (~60 times/sec at 60fps)
            // This ensures maximum throughput for R-NaD training.
            if (!_aiPending)
            {
                ScheduleAI();
            }
        } catch (System.Exception ex) {
            if (_processCount % 600 == 0) {
                Logger.Error($"[AutoAI] Error in _Process polling: {ex.Message}");
            }
        }
    }

    // Watchdog removed in favor of _Process high-frequency polling.


    // -----------------------------------------------------------------------
    // Hook registration
    // -----------------------------------------------------------------------


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