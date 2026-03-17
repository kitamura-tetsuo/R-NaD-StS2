using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.RestSite;

namespace communication_mod.Handlers;

public class AiRestSiteRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/RestSiteRoom";
    public RoomType[] HandledTypes => new[] { RoomType.RestSite };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for rest site room");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        NRestSiteRoom room = await WaitHelper.ForNode<NRestSiteRoom>(root, _roomPath, ct, (TimeSpan?)null);

        MainFile.Logger.Info("[AiSlayer] Handling rest site options via AI");
        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            await MainFile.Instance.StepAI();
            await Task.Delay(200, ct);

            // Break if proceed button is ready or an overlay screen opened (like card choice)
            if (room.ProceedButton.IsEnabled) break;
            if (MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance?.ScreenCount > 0) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
        }

        if (room.ProceedButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(room.ProceedButton);
        }
    }
}
