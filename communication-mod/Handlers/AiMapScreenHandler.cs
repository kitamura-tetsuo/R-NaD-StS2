using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace communication_mod.Handlers;

public class AiMapScreenHandler : IHandler
{
    public TimeSpan Timeout => TimeSpan.FromMinutes(2);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for map screen");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        MegaCrit.Sts2.Core.Nodes.NRun runNode = root.GetNode<MegaCrit.Sts2.Core.Nodes.NRun>(new NodePath("/root/Game/RootSceneContainer/Run"));
        
        await WaitHelper.Until(() => ((CanvasItem)runNode.GlobalUi.MapScreen).IsVisibleInTree(), ct, TimeSpan.FromSeconds(10), "Map screen not visible");

        MainFile.Logger.Info("[AiSlayer] Handling map navigation via AI");
        
        var startTime = DateTime.UtcNow;
        var maxWaitTime = TimeSpan.FromSeconds(60);
        int consecutiveNoActionCount = 0;
        const int maxNoActionRetries = 20;
        
        while (!ct.IsCancellationRequested)
        {
            // Check timeout
            if (DateTime.UtcNow - startTime > maxWaitTime)
            {
                MainFile.Logger.Warn("[AiSlayer] Map screen timeout - forcing map node selection");
                break;
            }

            // AI model will select a map node
            bool actionExecuted = await MainFile.Instance.StepAI(MainFile.Instance.ExecuteMapAction);
            
            if (!actionExecuted)
            {
                consecutiveNoActionCount++;
                MainFile.Logger.Info($"[AiSlayer] No action executed ({consecutiveNoActionCount}/{maxNoActionRetries})");
                
                if (consecutiveNoActionCount >= maxNoActionRetries)
                {
                    MainFile.Logger.Warn("[AiSlayer] Too many failed attempts - breaking out of map loop");
                    break;
                }
            }
            else
            {
                consecutiveNoActionCount = 0;
            }
            
            await Task.Delay(500, ct);

            // Check if we entered a room
            var runState = RunManager.Instance.DebugOnlyGetState();
            if (runState?.CurrentRoom != null && runState.CurrentRoom.RoomType != RoomType.Unassigned)
            {
                if (!runNode.GlobalUi.MapScreen.IsOpen) break;
            }
        }
    }
}
