using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
    private static bool _diagnosed = false;

    private static readonly System.Text.Json.JsonSerializerOptions JsonOptions = new()
    {
        IncludeFields = true,
        PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
    };

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
            if (AiBridge != null)
            {
                PollCommands();
                TryAiMove();
            }
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
            Logger.Info($"[AutoAI] NGame.Instance: {(ngame != null ? "exists" : "null")}");
            if (ngame != null)
            {
                var ironclad = ModelDb.Character<Ironclad>();
                var acts = ActModel.GetDefaultList();
                var modifiers = new List<ModifierModel>();
                
                Logger.Info($"[AutoAI] Launching new run with {ironclad.Id.Entry}...");
                
                // Call the deferred method to ensure it runs on the main thread
                CallDeferred(nameof(StartNewGameDeferred));
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

    private async void StartNewGameDeferred()
    {
        try
        {
            var ngame = MegaCrit.Sts2.Core.Nodes.NGame.Instance;
            if (ngame == null) return;

            var ironclad = ModelDb.Character<Ironclad>();
            var acts = ActModel.GetDefaultList();
            var modifiers = new List<ModifierModel>();

            // StartNewSingleplayerRun handles the main run initialization
            var runState = await ngame.StartNewSingleplayerRun(
                ironclad, 
                true, 
                acts, 
                modifiers, 
                "", // Use random seed
                0, 
                null
            );
            
            Logger.Info("[AutoAI] Run started. Waiting for scene to settle...");
            await Task.Delay(2000); // Wait for transition effects
            
            var rm = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            var state = rm.DebugOnlyGetState();
            
            if (state != null && state.CurrentRoom is MapRoom)
            {
                Logger.Info("[AutoAI] Landed in MapRoom. Entering starting combat node...");
                await rm.EnterMapCoord(state.Map.StartingMapPoint.coord);
                Logger.Info("[AutoAI] Successfully entered first node.");
            }
            else
            {
                string roomType = state?.CurrentRoom?.GetType().Name ?? "null";
                Logger.Info($"[AutoAI] Current room is {roomType}. No auto-map entry needed.");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Error in StartNewGameDeferred: {ex.Message}\n{ex.StackTrace}");
        }
    }

    private void TryAiMove()
    {
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        if (cm == null || !cm.IsInProgress || !cm.IsPlayPhase || cm.PlayerActionsDisabled) return;

        try
        {
            string stateJson = GetJsonState();
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
                    string action = dict["action"].AsString();
                    if (action == "play_card")
                    {
                        // TODO: Implement card playing logic
                        string cardId = dict["card_id"].AsString();
                        Logger.Info($"[AutoAI] AI wants to play card: {cardId}");
                    }
                    else if (action == "wait")
                    {
                        Logger.Info("[AutoAI] AI is waiting.");
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error in TryAiMove: {ex.Message}");
        }
    }

    private string GetJsonState()
    {
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        var combatState = cm.DebugOnlyGetState();
        if (combatState == null) return "{}";

        var player = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(combatState);
        if (player == null) return "{}";

        var pState = player.PlayerCombatState;
        
        var state = new
        {
            player = new
            {
                hp = player.Creature.CurrentHp,
                maxHp = player.Creature.MaxHp,
                block = player.Creature.Block,
                energy = pState?.Energy ?? 0,
                stars = pState?.Stars ?? 0
            },
            hand = pState?.Hand.Cards.Select(c => new
            {
                id = c.Id.Entry,
                name = c.Title,
                cost = c.EnergyCost.GetWithModifiers(MegaCrit.Sts2.Core.Entities.Cards.CostModifiers.All),
                starCost = c.GetStarCostWithModifiers(),
                isPlayable = c.CanPlay()
            }).ToList(),
            drawPileCount = pState?.DrawPile.Cards.Count ?? 0,
            discardPileCount = pState?.DiscardPile.Cards.Count ?? 0,
            exhaustPileCount = pState?.ExhaustPile.Cards.Count ?? 0,
            enemies = combatState.Enemies.Where(e => e.IsAlive).Select(e => new
            {
                id = e.ModelId.Entry,
                name = e.Name,
                hp = e.CurrentHp,
                maxHp = e.MaxHp,
                block = e.Block,
                intent = e.Monster?.NextMove.Intents.Select(i => new
                {
                    type = i.IntentType.ToString(),
                    damage = (i is MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent ai) ? ai.GetSingleDamage(combatState.PlayerCreatures, e) : 0,
                    hits = (i is MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent ai2) ? ai2.Repeats + 1 : 0
                }).ToList()
            }).ToList()
        };

        return System.Text.Json.JsonSerializer.Serialize(state, JsonOptions);
    }
}