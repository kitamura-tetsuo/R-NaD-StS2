using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;

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
        NRestSiteRoom room = await WaitHelper.ForNode<NRestSiteRoom>(root, _roomPath, ct, null);

        MainFile.Logger.Info("[AiSlayer] Handling rest site options via AI");
        while (!ct.IsCancellationRequested)
        {

            // AI decides which rest site option to click
            await MainFile.Instance.StepAI();
            await Task.Delay(500, ct);

            // Wait until proceed button is enabled or an overlay screen opened
            await WaitHelper.Until(delegate
            {
                if (room.ProceedButton.IsEnabled) return true;
                if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) return true;
                if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) return true;
                return false;
            }, ct, TimeSpan.FromSeconds(10), "Rest site option did not respond");

            if (room.ProceedButton.IsEnabled) break;
            if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
        }

        if (room.ProceedButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(room.ProceedButton);
        }
    }
}
