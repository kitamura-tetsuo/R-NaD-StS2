using Godot;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Text.Json;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Entities.Characters;
using Logger = MegaCrit.Sts2.Core.Logging.Logger;
using HarmonyLib;

namespace recorder_mod
{
    [ModInitializer(nameof(Initialize))]
    public partial class MainFile : Node
    {
        private static MainFile? _instance;
        public static MainFile Instance => _instance!;
        public static readonly Logger Logger = new Logger("RecorderMod", LogType.Generic);

        public bool CollectionMode { get; private set; } = false;
        private string _defaultSeed = "";
        private bool _gymMode = false;
        private DataCollector? _dataCollector;
        private Harmony? _harmony;
        private bool _isInitialized = false;
        private bool _isStartingRun = false;
        private long _lastPollTime = 0;

        public static readonly JsonSerializerOptions JsonOptions = new()
        {
            IncludeFields = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public MainFile()
        {
            _instance = this;
        }

        public override void _Ready()
        {
            // BaseLib calls static Initialize, which sets up the instance.
        }

        public static void Initialize()
        {
            if (_instance == null)
            {
                _instance = new MainFile();
            }
            _instance.CallDeferred(nameof(SafeSetup));
        }

        private void SafeSetup()
        {
            var tree = Engine.GetMainLoop() as SceneTree;
            if (tree == null || tree.Root == null)
            {
                this.GetTree()?.CreateTimer(0.1)
                    .Connect("timeout", new Callable(this, nameof(SafeSetup)));
                return;
            }

            if (this.GetParent() == null)
            {
                this.ProcessMode = ProcessModeEnum.Always;
                tree.Root.AddChild(this);
                this.SetProcess(true);
            }

            DoInitialize();
        }

        private void DoInitialize()
        {
            Logger.Info("[Recorder] MainFile.Initialize called");
            string[] args = OS.GetCmdlineArgs();
            bool collect = args.Contains("--collect");
            string defaultSeed = "1";
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "--seed")
                {
                    defaultSeed = args[i + 1];
                    break;
                }
            }

            CollectionMode = collect;
            _defaultSeed = defaultSeed;
            _dataCollector = new DataCollector();
            // Don't call Initialize here if it might fail due to null RunManager
            // Instead, do it in _Process
            Logger.Info($"[Recorder] Initialized. CollectionMode: {CollectionMode}, Seed: '{_defaultSeed}'");
            
            if (CollectionMode) {
                // Save a debug file to verify it's working
                try {
                    string debugInfo = $"RecorderMod Initialized at {DateTime.Now}\n" +
                                       $"UserDataDir: {OS.GetUserDataDir()}\n";
                    File.WriteAllText(Path.Combine(OS.GetUserDataDir(), "recorder_debug.txt"), debugInfo);
                } catch {}
            }

            // Apply Harmony patches
            try {
                _harmony = new Harmony("com.kitamura-tetsuo.recorder-mod");
                _harmony.PatchAll(Assembly.GetExecutingAssembly());
                
                Logger.Info("[Recorder] Harmony patches applied (standard only).");
            } catch (Exception e) {
                Logger.Error($"[Recorder] Harmony patch failed: {e.Message}");
            }
        }

        public override void _Process(double delta)
        {
            if (CollectionMode && _dataCollector != null) {
                if (!_dataCollector.IsInitialized) {
                    _dataCollector.Initialize();
                }
                _dataCollector.Update();
            }

            var rm = RunManager.Instance;
            if (rm == null) return;

            // Wait for MainMenu or RunInProgress
            if (rm.IsInProgress) {
                _isStartingRun = false;
                if (!_isInitialized) {
                    _isInitialized = true;
                    Logger.Info("[Recorder] Detected active run.");
                }
            } else {
                if (!_isStartingRun) {
                    Node rootNode = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
                    bool hasMenu = rootNode.GetNodeOrNull("Game/RootSceneContainer/MainMenu") != null;
                    if (hasMenu) {
                        _isStartingRun = true;
                        Logger.Info($"[Recorder] MainMenu detected. Starting seeded run with seed '{_defaultSeed}'...");
                        StartSts2Run(_defaultSeed);
                    } else {
                        long now = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                        if (now - _lastPollTime > 5000) {
                            Logger.Info("[Recorder] Waiting for MainMenu or Run state...");
                            _lastPollTime = now;
                        }
                    }
                }
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
            var ngame = NGame.Instance;
            if (ngame == null) return;

            var rm = RunManager.Instance;
            if (rm != null && rm.IsInProgress)
            {
                rm.CleanUp(false);
                await Task.Delay(100);
            }

            var ironclad = ModelDb.Character<Ironclad>();
            var acts     = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            Logger.Info($"[Recorder] Starting new seeded run: {seedToUse}");
            await ngame.StartNewSingleplayerRun(ironclad, true, acts, modifiers, seedToUse, 0, 0);
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
}
