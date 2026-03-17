using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace communication_mod.Handlers;

public class AiTreasureRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/TreasureRoom";
    public RoomType[] HandledTypes => new[] { RoomType.Treasure };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for treasure room");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        NTreasureRoom room = await WaitHelper.ForNode<NTreasureRoom>(root, _roomPath, ct, (TimeSpan?)null);

        // Open chest automatically if available, just like the original handler
        NClickableControl chest = ((Node)room).GetNodeOrNull<NClickableControl>("Chest");
        if (chest != null && chest.IsEnabled && ((CanvasItem)chest).Visible)
        {
            MainFile.Logger.Info("[AiSlayer] Opening chest");
            await UiHelper.Click(chest);
            await Task.Delay(1000, ct);
        }

        MainFile.Logger.Info("[AiSlayer] Handling relics/rewards via AI");
        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            // The AI model will decide to pick up relics
            await MainFile.Instance.StepAI();
            await Task.Delay(200, ct);

            // Exit loop if proceed button is ready or map is open
            if (room.ProceedButton.IsEnabled) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
        }

        if (room.ProceedButton.IsEnabled)
        {
            await WaitHelper.Until(() => room.ProceedButton.IsEnabled, ct, TimeSpan.FromSeconds(5L), "Proceed button not enabled");
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(room.ProceedButton);
        }
    }
}
