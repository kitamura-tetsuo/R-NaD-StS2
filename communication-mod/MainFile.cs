using Godot;

using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace communication_mod;

/// <summary>
/// Mod のエントリポイント。初期化・メインループのみを担当します。
/// </summary>
[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string ModId = "communication-mod";

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    public override void _Process(double delta)
    {
        _processTimer += delta;
        if (_processTimer > 2.0)
        {
            Logger.Info("[AutoAI] _Process active");
            _processTimer = 0;
        }

        PollBridge();
    }

    private void PollBridge()
    {
        if (AiBridge == null) return;
        
        try
        {
            string stateJson = GetJsonState();
            var responseVariant = AiBridge.Call("predict_action", stateJson);
            if (responseVariant.VariantType == Variant.Type.Nil) return;
            
            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action"))
                {
                    string action = dict["action"].AsString();
                    if (action == "command")
                    {
                        string command = dict["command"].AsString();
                        Logger.Info($"[AutoAI] Received command: {command}");
                        ProcessCommand(command);
                    }
                    else if (action != "wait" || stateJson != "{\"type\":\"none\"}")
                    {
                        ExecuteAction(dict);
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Error in PollBridge: {ex.Message}");
        }
    }

    private double _processTimer = 0;

    private static Node? _aiBridge;
    public static Node? AiBridge 
    { 
        get => _aiBridge;
        private set 
        {
            _aiBridge = value;
            if (_aiBridge != null)
            {
                InitializeHooks();
            }
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
        Engine.TimeScale = 10.0f;

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
                _instance?.GetTree()?.CreateTimer(1.0).Connect("timeout", new Callable(_instance, nameof(SafeSetup)));
                return;
            }

            if (_instance != null && _instance.GetParent() == null)
            {
                tree.Root.AddChild(_instance);
                Logger.Info("[AutoAI] MainFile added to SceneTree.");
            }

            string extensionPath = "res://ai_bridge.gdextension";
            GDExtensionManager.LoadExtension(extensionPath);

            AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
            if (AiBridge != null)
            {
                AiBridge.Name = "MyAiBridge";
                tree.Root.AddChild(AiBridge);
                Logger.Info("[AutoAI] Mod initialized and AiBridge attached.");
                _instance?.CallDeferred(nameof(PollBridge));
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Initialization error: {ex.Message}");
        }
    }

    private static void InitializeHooks()
    {
        try 
        {
            var combatManager = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
            if (combatManager != null)
            {
                combatManager.TurnStarted += (state) => _instance?.CallDeferred(nameof(PollBridge));
                combatManager.PlayerActionsDisabledChanged += (state) => {
                    if (!combatManager.PlayerActionsDisabled)
                        _instance?.CallDeferred(nameof(PollBridge));
                };
            }

            var overlayStack = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance;
            if (overlayStack != null)
            {
                overlayStack.ChildEnteredTree += (node) => _instance?.CallDeferred(nameof(PollBridge));
            }

            var runManager = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (runManager != null)
            {
                runManager.ActionExecutor.AfterActionExecuted += (action) => _instance?.CallDeferred(nameof(PollBridge));
            }
        }
        catch (System.Exception ex)
        {
             Logger.Warn($"[AutoAI] Hooks error: {ex.Message}");
        }
    }

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
            string path = command.Substring("screenshot:".Length);
            TakeScreenshot(path);
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
                Logger.Info("[AutoAI] Calling CallDeferred(StartSts2RunDeferred)");
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
            var acts = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            Logger.Info($"[AutoAI] Calling StartNewSingleplayerRun with seed: {seedToUse}");
            var runState = await ngame.StartNewSingleplayerRun(ironclad, true, acts, modifiers, seedToUse, 0, null);

            Logger.Info("[AutoAI] StartNewSingleplayerRun returned. Waiting 2s...");
            await Task.Delay(2000);
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
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
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] CRITICAL ERROR in StartSts2RunDeferred: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private void TakeScreenshot(string path)
    {
        Logger.Info($"[AutoAI] Taking screenshot to: {path}");
        try
        {
            var viewport = GetViewport();
            var texture = viewport?.GetTexture();
            var image = texture?.GetImage();
            if (image != null)
            {
                string dir = System.IO.Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(dir) && !System.IO.Directory.Exists(dir)) System.IO.Directory.CreateDirectory(dir);
                image.SavePng(path);
                Logger.Info($"[AutoAI] Screenshot saved.");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Screenshot error: {ex.Message}");
        }
    }
}