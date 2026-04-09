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
        private DataCollector? _dataCollector;
        private Harmony? _harmony;
        private bool _isInitialized = false;

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
            CollectionMode = true;
            _dataCollector = new DataCollector();
            Logger.Info("[Recorder] Initialized in passive mode.");

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
            if (_dataCollector != null) {
                if (!_dataCollector.IsInitialized) {
                    _dataCollector.Initialize();
                }
                _dataCollector.Update();
            }
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
