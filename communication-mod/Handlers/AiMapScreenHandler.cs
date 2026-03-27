// Reference: StS2_Decompiled/MegaCrit.Sts2.Core.AutoSlay.Handlers.Screens/MapScreenHandler.cs

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
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        MegaCrit.Sts2.Core.Nodes.NRun runNode = root.GetNode<MegaCrit.Sts2.Core.Nodes.NRun>(new NodePath("/root/Game/RootSceneContainer/Run"));

        int initialFloor = RunManager.Instance?.DebugOnlyGetState()?.TotalFloor ?? -1;

        // Wait for map screen to be open (not necessarily visible in tree, as animations might delay visibility)
        // OR if the floor already advanced (handling race condition where map was handled prematurely)
        await WaitHelper.Until(delegate {
            int currentFloor = RunManager.Instance?.DebugOnlyGetState()?.TotalFloor ?? -1;
            bool floorAdvanced = initialFloor != -1 && currentFloor > initialFloor;
            
            if (runNode.GlobalUi.MapScreen.IsOpen || floorAdvanced) return true;
            
            // If map is NOT open, check if there's an overlay blocking it and drain it
            if (MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance != null && 
                MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance.ScreenCount > 0)
            {
                MainFile.Logger.Info("[AiSlayer] Overlay detected while waiting for map, draining...");
                // Note: We can't await here directly in the predicate, but we can return true to break the wait
                // and then handle it in the loop below.
                return true; 
            }
            
            return false;
        }, ct, TimeSpan.FromSeconds(15), "Map screen not open");

        // Extra check: if we broke out of Until because of an overlay, drain it now
        if (!runNode.GlobalUi.MapScreen.IsOpen)
        {
             await MainFile.Instance.GetAiSlayer().DrainOverlayScreensAsync(ct);
             
             // After draining, re-check map (it might be open now)
             if (!runNode.GlobalUi.MapScreen.IsOpen) {
                  // Final short wait for map
                  await WaitHelper.Until(() => runNode.GlobalUi.MapScreen.IsOpen, ct, TimeSpan.FromSeconds(5), "Map screen still not open after draining overlays");
             }
        }

        if (runNode.GlobalUi.MapScreen.IsOpen)
        {
            MainFile.Logger.Info("[AiSlayer] Map screen open, handling navigation via AI");
        }
        else
        {
            MainFile.Logger.Info("[AiSlayer] Map screen not open and/or floor already advanced. Skipping map navigation.");
            return;
        }
        
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
