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
                    if (command.StartsWith("start_game"))
                    {
                        string seed = "";
                        if (command.Contains(":"))
                        {
                            seed = command.Split(':')[1];
                        }
                        StartNewGame(seed);
                    }
                    else if (command.StartsWith("screenshot:"))
                    {
                        string path = command.Substring("screenshot:".Length);
                        TakeScreenshot(path);
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Error polling commands: {ex.Message}");
        }
    }

    private void StartNewGame(string seed = "")
    {
        Logger.Info($"[AutoAI] Starting new game with seed: {seed}");
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
                // Pass seed via a temporary field or directly if we use a different pattern
                _pendingSeed = seed;
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

    private string _pendingSeed = "";

    private async void StartNewGameDeferred()
    {
        try
        {
            string seedToUse = _pendingSeed;
            _pendingSeed = ""; // Clear it
            
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
                seedToUse, // Use provided seed
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
    private void TakeScreenshot(string path)
    {
        Logger.Info($"[AutoAI] Taking Godot screenshot to: {path}");
        try
        {
            var viewport = GetViewport();
            if (viewport == null)
            {
                Logger.Error("[AutoAI] GetViewport() returned null.");
                return;
            }

            // Ensure we wait for the frame to render if needed, but in _Process it should be okay
            var texture = viewport.GetTexture();
            if (texture == null)
            {
                Logger.Error("[AutoAI] viewport.GetTexture() returned null.");
                return;
            }

            var image = texture.GetImage();
            if (image == null)
            {
                Logger.Error("[AutoAI] texture.GetImage() returned null.");
                return;
            }

            // Ensure directory exists
            string dir = System.IO.Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !System.IO.Directory.Exists(dir))
            {
                System.IO.Directory.CreateDirectory(dir);
            }

            Error err = image.SavePng(path);
            if (err == Error.Ok)
            {
                Logger.Info($"[AutoAI] Screenshot saved successfully to {path}");
            }
            else
            {
                Logger.Error($"[AutoAI] Failed to save screenshot. Error: {err}");
            }
        }
        catch (System.Exception ex)
        {
            Logger.Error($"[AutoAI] Error taking screenshot: {ex.Message}\n{ex.StackTrace}");
        }
    }
}
