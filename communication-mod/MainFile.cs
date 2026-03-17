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

    public static MainFile Instance { get; private set; }


    private static long _perfLogTimer = 0;
    private static System.Diagnostics.Stopwatch _frameStopwatch = new();
    private AiSlayer _aiSlayer;
    private long _lastActionTime = 0;
    public bool _gymMode = false;
    private bool _noSpeedup = false;
    private string _defaultSeed = "";
    private bool _isSteppingAI = false;

    private void ScheduleAI()
    {
        // No longer used, handled by AiSlayer
    }

    public static bool IsGameBusy()
    {
        var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
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
                if (!cm.IsPlayPhase) return true;
            }
        }

        return false;
    }


    public async Task StepAI()
    {
        if (AiBridge == null) return;
        if (_isSteppingAI) return;

        _isSteppingAI = true;
        try
        {
            string stateJson = GetJsonState();
            long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            var swBridge = System.Diagnostics.Stopwatch.StartNew();
            var responseVariant = AiBridge.Call("predict_action", stateJson);
            long bridgeTime = swBridge.ElapsedMilliseconds;
            
            if (bridgeTime > 500)
            {
                Logger.Warn($"[PERF] AiBridge.predict_action took {bridgeTime}ms");
            }

            if (responseVariant.VariantType == Variant.Type.Nil) return;

            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) != Error.Ok) return;

            var dict = json.Data.AsGodotDictionary();
            if (!dict.ContainsKey("action")) return;

            string action = dict["action"].AsString();
            
            // Minimum delay between actions (except wait/screenshot)
            if (currentTime - _lastActionTime < 50 && action != "wait" && action != "take_screenshot")
            {
                return;
            }
            _lastActionTime = currentTime;

            if (action == "wait") return;

            Logger.Info($"[AutoAI] StepAI: action={action}");

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
            }
            else
            {
                await ExecuteAction(dict);
            }
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] Error in StepAI: {ex.Message}");
        }
        finally
        {
            _isSteppingAI = false;
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
        bool noSpeedup = false;
        string defaultSeed = "";
        for (int i = 0; i < args.Length; i++) {
            if (args[i] == "--gym" || args[i] == "gym") gym = true;
            if (args[i] == "--no-speedup") noSpeedup = true;
            if (args[i] == "--seed" && i + 1 < args.Length) defaultSeed = args[i + 1];
        }

        if (gym && !noSpeedup)
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
        Instance = _instance;
        Instance.Name = "R_NaD_Controller";
        Instance._gymMode = gym;
        Instance._noSpeedup = noSpeedup;
        Instance._defaultSeed = defaultSeed;
        Instance._aiSlayer = new AiSlayer();
        Instance?.CallDeferred(nameof(SafeSetup));
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

            var sw = System.Diagnostics.Stopwatch.StartNew();
            GDExtensionManager.LoadExtension("res://ai_bridge.gdextension");
            Logger.Info($"[PERF] LoadExtension took {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
            Logger.Info($"[PERF] Instantiate AiBridge took {sw.ElapsedMilliseconds}ms");

            if (AiBridge != null)
            {
                AiBridge.Name = "MyAiBridge";
                tree.Root.AddChild(AiBridge);
                
                sw.Restart();
                // Trigger bridge initialization (and thus HTTP server start)
                if (Instance != null) Instance.PollScreenshotRequest();
                Logger.Info("[PERF] Dispatched initial PollScreenshotRequest");

                if (Instance != null && Instance._gymMode)
                {
                    Instance._aiSlayer.Start(Instance._defaultSeed);
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] SafeSetup error: {ex.Message}");
        }
    }
    private bool _bridgeInitialized = false;

    public override void _Process(double delta)
    {
        _frameStopwatch.Restart();
        long now = DateTimeOffset.Now.ToUnixTimeMilliseconds();

        // Performance monitoring every 5 seconds
        if (now - _perfLogTimer > 5000)
        {
            _perfLogTimer = now;
            double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
            double process = Performance.GetMonitor(Performance.Monitor.TimeProcess) * 1000.0;
            double physics = Performance.GetMonitor(Performance.Monitor.TimePhysicsProcess) * 1000.0;
            double static_mem = Performance.GetMonitor(Performance.Monitor.MemoryStatic) / (1024.0 * 1024.0);
            double obj_count = Performance.GetMonitor(Performance.Monitor.ObjectCount);
            
            Logger.Info($"[PERF] FPS: {fps:F1}, Process: {process:F2}ms, Phys: {physics:F2}ms, Mem: {static_mem:F1}MB, Objs: {obj_count}");
        }

        if (_gymMode && !_noSpeedup)
        {
            if (SaveManager.Instance != null && SaveManager.Instance.PrefsSave != null)
            {
                SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
            }
        }

        if (AiBridge == null) return;

        // Delay bridge interactions until game is loaded to avoid initial startup freeze
        if (!_bridgeInitialized)
        {
            // Only start bridge if we are in MainMenu or beyond, 
            // and we've waited at least 5 seconds from boot
            if (now - _perfLogTimer < 0) return; // Wait at least one timer cycle (redundant)
            
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (rm == null) return;
            
            _bridgeInitialized = true;
            Logger.Info("[AutoAI] Bridge initialized - beginning polling.");
        }

        // Even if AiSlayer is running its own loop, we poll StepAI periodically
        // to handle background commands (like start_game) and keep the server alive/responsive.
        long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        if (currentTime - _lastActionTime > 500) // Poll every 500ms when idle
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            // Poll for screenshot requests from the bridge
            PollScreenshotRequest();
            long screenshotTime = sw.ElapsedMilliseconds;

            if (!_isSteppingAI)
            {
                _ = StepAI(); 
            }
            long totalTime = sw.ElapsedMilliseconds;
            
            if (totalTime > 100) {
                 Logger.Info($"[PERF] StepAI/PollScreenshot took {totalTime}ms (Bridge: {totalTime-screenshotTime}ms)");
            }
        }
        
        long frameFinal = _frameStopwatch.ElapsedMilliseconds;
        if (frameFinal > 32) {
             Logger.Warn($"[PERF] _Process took {frameFinal}ms");
        }
    }

    private void PollScreenshotRequest()
    {
        if (AiBridge == null) return;
        
        var pathVariant = AiBridge.Call("check_screenshot_request");
        if (pathVariant.VariantType == Variant.Type.String)
        {
            string path = pathVariant.AsString();
            if (!string.IsNullOrEmpty(path))
            {
                Logger.Info($"[AutoAI] Taking requested screenshot: {path}");
                TakeScreenshot(path);
                AiBridge.Call("mark_screenshot_done");
            }
        }
    }

    private void TakeScreenshotDeferred(string path)
    {
        TakeScreenshot(path);
    }

    private void ProcessCommand(string command)
    {
        if (command.StartsWith("start_game"))
        {
            string seed = command.Contains(":") ? command.Split(':')[1] : "";
            if (_aiSlayer != null) _aiSlayer.Start(seed);
            else StartSts2Run(seed);
        }
    }

    public void StartSts2Run(string seed = "")
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