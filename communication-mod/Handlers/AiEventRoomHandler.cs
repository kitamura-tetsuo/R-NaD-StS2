using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace communication_mod.Handlers;

public class AiEventRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/EventRoom";
    public RoomType[] HandledTypes => new[] { RoomType.Event };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for event room");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        Node eventRoom = await WaitHelper.ForNode<Node>(root, _roomPath, ct, (TimeSpan?)null);

        MainFile.Logger.Info("[AiSlayer] Handling event via AI");
        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            // AI decides which event option to click
            await MainFile.Instance.StepAI();
            await Task.Delay(200, ct);

            // Exit if map is open or room no longer valid (e.g. combat started)
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
            if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree()) break;
        }
    }
}
