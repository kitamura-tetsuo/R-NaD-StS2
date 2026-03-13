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

    public static Node? AiBridge { get; private set; }
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