using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;
using System.Linq;

namespace communication_mod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string
        ModId = "communication-mod"; //At the moment, this is used only for the Logger and harmony names.

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    public static Node? AiBridge { get; private set; }
    private static MainFile? _instance;

    public static void Initialize()
    {
        Harmony harmony = new(ModId);
        harmony.PatchAll();

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
            PollCommands();
            TryAiMove();
        }
    }

    private void PollCommands()
    {
        if (AiBridge == null) return;

        try
        {
            // Send empty state to just poll for commands
            var responseVariant = AiBridge.Call("predict_action", "{}");
            string response = responseVariant.AsString();
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action") && dict["action"].AsString() == "command")
                {
                    string command = dict["command"].AsString();
                    Logger.Info($"[AutoAI] Received command: {command}");
                    if (command == "start_game")
                    {
                        StartNewGame();
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error polling commands: {ex.Message}");
        }
    }

    private void StartNewGame()
    {
        Logger.Info("[AutoAI] Starting new game...");
        try
        {
            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            if (ngame != null)
            {
                // Correct signature for StS2: (CharacterModel, bool, IReadOnlyList<ActModel>, IReadOnlyList<ModifierModel>, string, int, DateTimeOffset?)
                ngame.StartNewSingleplayerRun(null, true, null, null, null, 0, null);
            }
            else
            {
                Logger.Error("[AutoAI] NGame.Instance is null, cannot start game.");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error starting new game: {ex.Message}");
        }
    }

    private void TryAiMove()
    {
        /* Temporary commented out due to API mismatch with current sts2.dll
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        if (cm != null && cm.IsPlayPhase && !cm.PlayerActionsDisabled)
        {
            var player = cm.State?.Players.FirstOrDefault();
            var hand = player?.PlayerCombatState?.Hand?.Cards;
            if (hand == null || hand.Count == 0) return;

            // Simple serialization
            var handList = new Godot.Collections.Array();
            foreach (var card in hand)
            {
                handList.Add(card.Id.Entry);
            }

            var stateDict = new Godot.Collections.Dictionary
            {
                { "hp", player!.Creature.CurrentHp },
                { "energy", player.PlayerCombatState.Energy },
                { "hand", handList }
            };

            string jsonState = Json.Stringify(stateDict);

            if (AiBridge == null) return;
            var responseVariant = AiBridge.Call("predict_action", jsonState);
            string response = responseVariant.AsString();
            
            if (string.IsNullOrEmpty(response)) return;

            var json = new Json();
            if (json.Parse(response) == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action") && dict["action"].AsString() == "play_card")
                {
                    string cardId = dict["card_id"].AsString();
                    var cardToPlay = hand.FirstOrDefault(c => c.Id.Entry == cardId);
                    if (cardToPlay != null)
                    {
                        Logger.Info($"[AutoAI] AI chose to play: {cardId}");
                        CardSelectionPatch.IsAiPlaying = true;
                        cardToPlay.TryManualPlay(null);
                        CardSelectionPatch.IsAiPlaying = false;
                    }
                }
            }
        }
        */
    }
}