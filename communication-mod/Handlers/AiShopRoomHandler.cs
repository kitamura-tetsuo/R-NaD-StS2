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

namespace communication_mod.Handlers;

public class AiShopRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/MerchantRoom";
    public RoomType[] HandledTypes => new[] { RoomType.Shop };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for shop room");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        NMerchantRoom room = await WaitHelper.ForNode<NMerchantRoom>(root, _roomPath, ct, (TimeSpan?)null);

        MainFile.Logger.Info("[AiSlayer] Opening merchant inventory");
        room.OpenInventory();
        await Task.Delay(500, ct);

        MainFile.Logger.Info("[AiSlayer] Handling shopping via AI");
        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            // AI decides what to buy or when to leave
            await MainFile.Instance.StepAI();
            await Task.Delay(200, ct);

            // Exit loop if proceed button is ready or map is open
            if (room.ProceedButton.IsEnabled) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
        }

        if (room.ProceedButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(room.ProceedButton);
        }
    }
}
