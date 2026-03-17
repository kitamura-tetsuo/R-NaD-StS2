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
        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            // AI model will select a map node
            await MainFile.Instance.StepAI();
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
