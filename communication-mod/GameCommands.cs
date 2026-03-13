using Godot;

using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Rooms;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace communication_mod;

public partial class MainFile : Node
{
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
}
