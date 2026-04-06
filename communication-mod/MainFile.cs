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
using System.Linq;
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
    public AiSlayer GetAiSlayer() => _aiSlayer;
    private long _lastActionTime = 0;
    private string _defaultSeed = "";
    private long _lastPollTime = 0;
    private bool _isSteppingAI = false;
    private bool _gymMode = false;
    private bool _noSpeedup = false;
    private bool _isStartingRun = false;
    public bool TrainMode { get; private set; } = false;
    private long _lastIdleStepTime = 0;


    private void ScheduleAI()
    {
        // No longer used, handled by AiSlayer
    }




    public async Task<Variant> CallBridgeSafe(string method, Variant arg = default)
    {
        // If we're already on the main thread, call directly
        if (System.Threading.Thread.CurrentThread.ManagedThreadId == 1) // Heuristic for main thread in Godot
        {
            if (arg.VariantType == Variant.Type.Nil) return AiBridge?.Call(method) ?? default;
            return AiBridge?.Call(method, arg) ?? default;
        }

        var tcs = new TaskCompletionSource<Variant>();
        // Use a closure with Callable to move to the main thread safely
        Callable.From(() => {
            try {
                Variant result;
                if (arg.VariantType == Variant.Type.Nil) result = AiBridge?.Call(method) ?? default;
                else result = AiBridge?.Call(method, arg) ?? default;
                tcs.SetResult(result);
            } catch (Exception ex) {
                tcs.SetException(ex);
            }
        }).CallDeferred();
        
        return await tcs.Task;
    }

    public async Task<bool> StepAI(Func<Godot.Collections.Dictionary, Task> actionCallback)
    {
        if (AiBridge == null)
        {
            Logger.Warn("[AutoAI] StepAI: AiBridge is null");
            return false;
        }
        if (_isSteppingAI)
        {
            Logger.Warn("[AutoAI] StepAI: Already stepping, skipping");
            return false;
        }

        _isSteppingAI = true;
        try
        {
            string stateJson = GetJsonState();
            if (string.IsNullOrEmpty(stateJson)) return false;

            long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            var swBridge = System.Diagnostics.Stopwatch.StartNew();
            // Use thread-safe wrapper
            var responseVariant = await CallBridgeSafe("predict_action", stateJson);
            long bridgeTime = swBridge.ElapsedMilliseconds;
            
            if (bridgeTime > 500)
            {
                Logger.Warn($"[PERF] AiBridge.predict_action took {bridgeTime}ms");
            }

            if (responseVariant.VariantType == Variant.Type.Nil)
            {
                Logger.Warn("[AutoAI] StepAI: Response is nil");
                return false;
            }

            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response))
            {
                Logger.Warn("[AutoAI] StepAI: Response is empty");
                return false;
            }

            var json = new Json();
            if (json.Parse(response) != Error.Ok)
            {
                Logger.Warn($"[AutoAI] StepAI: JSON parse failed: {response}");
                return false;
            }

            var dict = json.Data.AsGodotDictionary();
            if (!dict.ContainsKey("action"))
            {
                Logger.Warn($"[AutoAI] StepAI: No action key in response: {response}");
                return false;
            }

            string action = dict["action"].AsString();
            
            if (action == "end_turn")
            {
                _waitingForVerification = true;
                Logger.Info("[AutoAI] StepAI: end_turn detected. Verification enabled for next turn.");
            }

            _lastActionTime = currentTime;

            if (action == "wait")
            {
                Logger.Info("[AutoAI] StepAI: Action is wait");
                return false;
            }

            Logger.Info($"[AutoAI] StepAI: action={action}");

            if (action == "command")
            {
                // Background commands are now handled by PollCommands()
                Logger.Info("[AutoAI] StepAI: Action is command, handled by PollCommands");
                return false;
            }
            else if (action == "take_screenshot")
            {
                // Screenshot requests are now handled by PollScreenshotRequest()
                Logger.Info("[AutoAI] StepAI: Action is take_screenshot, handled by PollScreenshotRequest");
                return false;
            }
            else
            {
                await actionCallback(dict);
                return true;
            }
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] Error in StepAI: {ex.Message}");
            return false;
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

    public static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new()
    {
        IncludeFields = true,
        PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
    };

    public static void Initialize()
    {
        Logger.Info("[AutoAI] MainFile.Initialize called");

        string[] args = OS.GetCmdlineArgs();
        Logger.Info($"[AutoAI] Command line args: {string.Join(", ", args)}");

        bool gym = args.Contains("--gym") || args.Contains("gym");
        bool noSpeedup = args.Contains("--no-speedup");
        bool train = args.Contains("--train") || args.Contains("train");
        bool offline = args.Contains("--offline") || System.Environment.GetEnvironmentVariable("RNAD_OFFLINE") == "true";
        bool collect = false;
        string defaultSeed = "";
        for (int i = 0; i < args.Length; i++) {
            if (args[i] == "--collect") collect = true;
            if (args[i] == "--seed" && i + 1 < args.Length) defaultSeed = args[i + 1];
        }

        Logger.Info($"[AutoAI] Flags: gym={gym}, noSpeedup={noSpeedup}, offline={offline}, collect={collect} (RN_OFF={System.Environment.GetEnvironmentVariable("RNAD_OFFLINE")})");

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
        Instance.TrainMode = train;
        Instance._defaultSeed = defaultSeed;
        Logger.Info($"[AutoAI] Initialized with defaultSeed: '{defaultSeed}'");
        
        // Debug file saving
        try {
            string debugInfo = $"Initialized at {DateTime.Now}\n" +
                               $"Seed: {defaultSeed}\n" +
                               $"UserDataDir: {OS.GetUserDataDir()}\n" +
                               $"ExecPath: {OS.GetExecutablePath()}\n";
            
            string userDebugPath = Path.Combine(OS.GetUserDataDir(), "mod_debug.txt");
            if (!Directory.Exists(OS.GetUserDataDir())) Directory.CreateDirectory(OS.GetUserDataDir());
            File.WriteAllText(userDebugPath, debugInfo);
            Logger.Info($"[AutoAI] Debug file saved to {userDebugPath}");
        } catch (Exception e) {
            Logger.Error($"[AutoAI] Failed to save debug file: {e.Message}");
        }

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
        long currentTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        if (currentTime - _lastPollTime > 125) // Poll every 125ms (Reduced from 500ms)
        {
            _lastPollTime = currentTime;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            // Poll for screenshot requests from the bridge
            _ = PollScreenshotRequest();
            // Poll for background commands (like start_game)
            _ = PollCommands();
            long pollTime = sw.ElapsedMilliseconds;
            
            if (pollTime > 100) {
                 Logger.Info($"[PERF] Background polling took {pollTime}ms");
            }

            // Periodic State Reporting (Idle Phase / Main Menu)
            // This ensures the bridge receives 'can_continue' status before a run starts.
            if (!AiSlayer.IsActive && currentTime - _lastIdleStepTime > 500)
            {
                _lastIdleStepTime = currentTime;
                // Handle actions (like Proceed/MainMenu) even in idle state to prevent hangs
                _ = StepAI(ExecuteUniversalAction); 
            }
        }
        
        long frameFinal = _frameStopwatch.ElapsedMilliseconds;
        if (frameFinal > 32) {
             Logger.Warn($"[PERF] _Process took {frameFinal}ms");
        }
    }

    private async Task PollScreenshotRequest()
    {
        if (AiBridge == null) return;
        
        var pathVariant = await CallBridgeSafe("check_screenshot_request");
        if (pathVariant.VariantType == Variant.Type.String)
        {
            string path = pathVariant.AsString();
            if (!string.IsNullOrEmpty(path))
            {
                Logger.Info($"[AutoAI] Taking requested screenshot: {path}");
                TakeScreenshot(path);
                await CallBridgeSafe("mark_screenshot_done");
            }
        }
    }

    private async Task PollCommands()
    {
        if (AiBridge == null) return;

        var responseVariant = await CallBridgeSafe("check_commands");
        if (responseVariant.VariantType == Variant.Type.Nil) return;

        string response = responseVariant.AsString();
        if (string.IsNullOrEmpty(response)) return;

        var json = new Json();
        if (json.Parse(response) != Error.Ok) return;

        var dict = json.Data.AsGodotDictionary();
        if (!dict.ContainsKey("action")) return;

        string action = dict["action"].AsString();
        if (action == "command")
        {
            string command = dict["command"].AsString();
            Logger.Info($"[AutoAI] Executing command: {command}");
            if (command == "start_game")
            {
                string seed = dict.ContainsKey("seed") ? dict["seed"].AsString() : "";
                StartSts2Run(seed);
            }
            else if (command == "continue_game")
            {
                ContinueSts2Run();
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
            StartSts2Run(seed);
        }
    }

    public void StartSts2Run(string seed = "")
    {
        var ngame = NGame.Instance;
        if (ngame != null)
        {
            string seedToUse = string.IsNullOrEmpty(seed) ? _defaultSeed : seed;
            CallDeferred(nameof(StartSts2RunDeferred), seedToUse);
        }
    }

    private async void StartSts2RunDeferred(string seedToUse)
    {
        try
        {
            if (_isStartingRun)
            {
                Logger.Info("[AutoAI] StartSts2RunDeferred: Already starting a run, skipping duplicate request.");
                return;
            }

            // Stop any active AI loop before cleanup
            _aiSlayer?.Stop();

            var ngame = NGame.Instance;
            if (ngame == null) return;

            var rm = RunManager.Instance;
            if (rm != null && rm.IsInProgress)
            {
                Logger.Info("[AutoAI] Cleaning up existing run before starting new one...");
                rm.CleanUp(false);
                await Task.Delay(200);
            }

            // Also check for Game Over screen that might be blocking the start
            var topOverlay = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance?.Peek();
            if (topOverlay is MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen.NGameOverScreen gos)
            {
                Logger.Info("[AutoAI] Game Over screen detected during setup. Closing it...");
                if (MegaCrit.Sts2.Core.Nodes.NGame.Instance != null)
                {
                    MegaCrit.Sts2.Core.Nodes.NGame.Instance.ReturnToMainMenu();
                    await Task.Delay(500);
                }
            }

            _isStartingRun = true;
            
            var ironclad = ModelDb.Character<Ironclad>();
            var acts     = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            Logger.Info($"[AutoAI] Starting new game with seed: '{seedToUse}'...");
            // Use 0 for the 7th argument as the method signature in this build expects an int
            await ngame.StartNewSingleplayerRun(ironclad, true, acts, modifiers, seedToUse, 0, 0);
            Logger.Info("[AutoAI] Run initialized. Waiting for state to settle...");
            await Task.Delay(1000);

            var state = rm?.DebugOnlyGetState();
            if (state?.CurrentRoom is MapRoom) await rm.EnterMapCoord(state.Map.StartingMapPoint.coord);
            
            Logger.Info($"[AutoAI] Run initialized. Starting AI loop with seed: {seedToUse}");
            _aiSlayer?.Start(seedToUse);
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] StartSts2RunDeferred error: {ex.Message}");
        }
        finally
        {
            _isStartingRun = false;
        }
    }

    public void ContinueSts2Run()
    {
        var ngame = NGame.Instance;
        if (ngame != null)
        {
            CallDeferred(nameof(ContinueSts2RunDeferred));
        }
    }

    private async void ContinueSts2RunDeferred()
    {
        try
        {
            if (_isStartingRun)
            {
                Logger.Info("[AutoAI] ContinueSts2RunDeferred: Already starting a run, skipping duplicate request.");
                return;
            }

            _aiSlayer?.Stop();

            var ngame = NGame.Instance;
            if (ngame == null) return;

            if (!SaveManager.Instance.HasRunSave)
            {
                Logger.Error("[AutoAI] ContinueSts2RunDeferred: No save file found.");
                return;
            }

            _isStartingRun = true;

            var saveData = SaveManager.Instance.LoadRunSave();
            if (saveData == null || !saveData.Success || saveData.SaveData == null)
            {
                Logger.Error("[AutoAI] ContinueSts2RunDeferred: Failed to load save.");
                return;
            }

            var serializableRun = saveData.SaveData;
            var runState = RunState.FromSerializable(serializableRun);
            RunManager.Instance.SetUpSavedSinglePlayer(runState, serializableRun);

            Logger.Info($"[AutoAI] Continuing run with character: {serializableRun.Players[0].CharacterId}");

            NGame.Instance.ReactionContainer.InitializeNetworking(new MegaCrit.Sts2.Core.Multiplayer.NetSingleplayerGameService());
            await NGame.Instance.LoadRun(runState, serializableRun.PreFinishedRoom);

            Logger.Info("[AutoAI] Run resumed. Starting AI loop.");
            _aiSlayer?.Start("");
        }
        catch (Exception ex)
        {
            Logger.Error($"[AutoAI] ContinueSts2RunDeferred error: {ex.Message}");
        }
        finally
        {
            _isStartingRun = false;
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
