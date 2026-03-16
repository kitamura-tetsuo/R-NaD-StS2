using Godot;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Reflection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Runs;
using System;
using System.Threading;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;

namespace communication_mod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string ModId = "communication-mod";

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    private bool _aiPending = false;
    private long _lastActionTime = 0;
    private int _lastTurnProcessed = -1;
    private bool _endTurnSentThisTurn = false;
    private bool _gymMode = false;
    private long _mainMenuWaitStartTime = 0;
    private string _lastStateJson = "";
    private long _lastPollTime = 0;
    private long _lastEndTurnTime = 0;
    private string _defaultSeed = "";

    private void ScheduleAI()
    {
        if (AiBridge == null) return;
        if (_aiPending) return;
        _aiPending = true;
        CallDeferred(nameof(TriggerAI));
    }

    private bool IsGameBusy()
    {
        var rm = RunManager.Instance;
        if (rm == null) return false;

        var runState = rm.DebugOnlyGetState();
        if (runState == null) return false;

        // Action queue is processing
        if (rm.ActionQueueSet != null && !rm.ActionQueueSet.IsEmpty)
        {
            return true;
        }

        // Combat animations or play phase not ready
        if (runState.CurrentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom)
        {
            var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            if (cm != null && cm.IsInProgress)
            {
                if (!cm.IsPlayPhase || cm.PlayerActionsDisabled)
                {
                    return true;
                }
            }
        }

        return false;
    }


    private void TriggerAI()
    {
        _aiPending = false;
        if (AiBridge == null) return;

        // Skip if playability is not met
        if (IsGameBusy())
        {
            return;
        }

        try
        {
            string stateJson = GetJsonState();
            long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            // Skip polling if state hasn't changed, unless 200ms passed (heartbeat)
            _lastPollTime = currentTime;
            _lastStateJson = stateJson;

            var responseVariant = AiBridge.Call("predict_action", stateJson);
            if (responseVariant.VariantType == Variant.Type.Nil) return;

            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) != Error.Ok) return;

            var dict = json.Data.AsGodotDictionary();
            if (!dict.ContainsKey("action")) return;

            string action = dict["action"].AsString();
            
            // Auto-start if stuck at main menu in gym mode
            if (_gymMode && action == "wait" && RunManager.Instance?.DebugOnlyGetState()?.CurrentRoom == null)
            {
                if (_mainMenuWaitStartTime == 0) _mainMenuWaitStartTime = currentTime;
                if (currentTime - _mainMenuWaitStartTime > 5000)
                {
                    Logger.Info("[AutoAI] Gym Mode auto-start triggered.");
                    _mainMenuWaitStartTime = currentTime;
                    StartSts2Run();
                    return;
                }
            }
            else
            {
                _mainMenuWaitStartTime = 0;
            }

            // Minimum delay between actions (except wait/screenshot)
            if (currentTime - _lastActionTime < 100 && action != "wait" && action != "take_screenshot")
            {
                return;
            }
            _lastActionTime = currentTime;

            if (action == "wait") return;

            Logger.Info($"[AutoAI] TriggerAI: action={action}");

            if (action == "command")
            {
                string command = dict["command"].AsString();
                ProcessCommand(command);
            }
            else if (action == "take_screenshot")
            {
                string path = dict["path"].AsString();
                TakeScreenshot(path);
                AiBridge.Call("mark_screenshot_done");
                ScheduleAI();
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

                // If we already sent it this turn, only allow re-sending after 2 seconds
                // This handles cases where the game missed the input or is lagging.
                if (_endTurnSentThisTurn && (currentTime - _lastEndTurnTime < 2000)) 
                {
                    return;
                }

                _endTurnSentThisTurn = true;
                _lastEndTurnTime = currentTime;
                ExecuteAction(dict);
            }
            else
            {
                ExecuteAction(dict);
            }
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] Error in TriggerAI: {ex.Message}");
        }
    }

    private static Node? _aiBridge;
    public static Node? AiBridge
    {
        get => _aiBridge;
        private set => _aiBridge = value;
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

        string[] args = OS.GetCmdlineArgs();
        Logger.Info($"[AutoAI] Command line args: {string.Join(", ", args)}");

        bool gym = false;
        string defaultSeed = "";
        for (int i = 0; i < args.Length; i++) {
            if (args[i] == "--gym" || args[i] == "gym") gym = true;
            if (args[i] == "--seed" && i + 1 < args.Length) defaultSeed = args[i + 1];
        }

        if (gym)
        {
            Logger.Info("[AutoAI] GYM MODE DETECTED: Initialization acceleration (non-Harmony approach).");
            try
            {
                PreloadManager.Enabled = false;
                NonInteractiveMode.AutoSlayerCheck = () => true;
                Logger.Info("[AutoAI] Basic acceleration parameters set.");
            }
            catch (Exception ex)
            {
                Logger.Error($"[AutoAI] Acceleration initialization failed: {ex.Message}");
            }
        }

        _instance = new MainFile();
        _instance.Name = "R_NaD_Controller";
        _instance._gymMode = gym;
        _instance._defaultSeed = defaultSeed;
        _instance?.CallDeferred(nameof(SafeSetup));
    }

    private static void SafeSetup()
    {
        try
        {
            SceneTree tree = (SceneTree)Engine.GetMainLoop();
            if (tree == null || tree.Root == null)
            {
                _instance?.GetTree()?.CreateTimer(0.1)
                    .Connect("timeout", new Callable(_instance, nameof(SafeSetup)));
                return;
            }

            if (_instance != null && _instance.GetParent() == null)
            {
                _instance.ProcessMode = ProcessModeEnum.Always;
                tree.Root.AddChild(_instance);
                _instance.SetProcess(true);
            }

            GDExtensionManager.LoadExtension("res://ai_bridge.gdextension");

            AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
            if (AiBridge != null)
            {
                AiBridge.Name = "MyAiBridge";
                tree.Root.AddChild(AiBridge);
                _instance?.CallDeferred(nameof(TriggerAI));
            }
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] SafeSetup error: {ex.Message}");
        }
    }

    public override void _Process(double delta)
    {
        if (_gymMode)
        {
            if (SaveManager.Instance != null && SaveManager.Instance.PrefsSave != null)
            {
                SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
            }
        }

        if (AiBridge == null) return;
        ScheduleAI();
    }

    private void ProcessCommand(string command)
    {
        if (command.StartsWith("start_game"))
        {
            string seed = command.Contains(":") ? command.Split(':')[1] : "";
            StartSts2Run(seed);
        }
    }

    private void StartSts2Run(string seed = "")
    {
        var ngame = NGame.Instance;
        if (ngame != null)
        {
            _pendingSeed = string.IsNullOrEmpty(seed) ? _defaultSeed : seed;
            CallDeferred(nameof(StartSts2RunDeferred));
        }
    }

    private string _pendingSeed = "";

    private async void StartSts2RunDeferred()
    {
        try
        {
            string seedToUse = _pendingSeed;
            _pendingSeed = "";
            var ngame = NGame.Instance;
            if (ngame == null) return;

            var ironclad = ModelDb.Character<Ironclad>();
            var acts     = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            Logger.Info("[AutoAI] Starting new game...");
            await ngame.StartNewSingleplayerRun(ironclad, true, acts, modifiers, seedToUse, 0, null);
            await Task.Delay(500);

            var rm    = RunManager.Instance;
            var state = rm.DebugOnlyGetState();
            if (state?.CurrentRoom is MapRoom) await rm.EnterMapCoord(state.Map.StartingMapPoint.coord);
            else ScheduleAI();
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] StartSts2RunDeferred error: {ex.Message}");
        }
    }

    private void TakeScreenshot(string path)
    {
        try
        {
            var image = GetViewport()?.GetTexture()?.GetImage();
            if (image != null)
            {
                string? dir = System.IO.Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(dir) && !System.IO.Directory.Exists(dir))
                    System.IO.Directory.CreateDirectory(dir);
                image.SavePng(path);
            }
        }
        catch (Exception ex) { Logger.Error($"[AutoAI] Screenshot error: {ex.Message}"); }
    }
}