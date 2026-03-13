using Godot;

using MegaCrit.Sts2.Core.Modding;

namespace communication_mod;

/// <summary>
/// Mod のエントリポイント。初期化・メインループのみを担当します。
/// 各機能は以下のファイルに分割されています:
///   - GameCommands.cs   : PollCommands / StartNewGame / StartNewGameDeferred
///   - ActionExecutor.cs : ExecuteAction と各アクションハンドラ
///   - GameStateProvider.cs : GetJsonState / GetMapJson
///   - Utilities.cs      : FindNodesByType など汎用ユーティリティ
/// </summary>
[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string ModId = "communication-mod";

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

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

    private static void InitializeHooks()
    {
        Logger.Info("[AutoAI] Initializing event-based hooks");
        
        // Combat hooks
        MegaCrit.Sts2.Core.Combat.CombatManager.Instance.TurnStarted += (state) => TriggerAiMove("Combat Turn Started");
        MegaCrit.Sts2.Core.Combat.CombatManager.Instance.PlayerActionsDisabledChanged += (state) => {
            if (!MegaCrit.Sts2.Core.Combat.CombatManager.Instance.PlayerActionsDisabled)
                TriggerAiMove("Player Actions Enabled");
        };

        // Screen/Overlay hooks
        MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance.ChildEnteredTree += (node) => TriggerAiMove($"Overlay Opened: {node.Name}");

        // Action completion hook
        MegaCrit.Sts2.Core.Runs.RunManager.Instance.ActionExecutor.AfterActionExecuted += (action) => TriggerAiMove("Action Executed");
    }

    private static void TriggerAiMove(string reason)
    {
        Logger.Info($"[AutoAI] Triggering AI move due to: {reason}");
        _instance?.CallDeferred(nameof(TryAiMove));
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
}