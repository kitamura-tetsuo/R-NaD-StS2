using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;

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
        NTreasureRoom room = await WaitHelper.ForNode<NTreasureRoom>(root, _roomPath, ct, null);

        NClickableControl chest = room.GetNode<NClickableControl>(new NodePath("Chest"));
        if (chest != null && chest.IsEnabled && chest.Visible)
        {
            MainFile.Logger.Info("[AiSlayer] Opening chest");
            await UiHelper.Click(chest);
            await Task.Delay(1000, ct);
        }

        MainFile.Logger.Info("[AiSlayer] Handling treasures via AI");
        int attempts = 0;
        int maxAttempts = 20;

        while (attempts < maxAttempts)
        {
            ct.ThrowIfCancellationRequested();


            // AI decides which relic to pick up
            await MainFile.Instance.StepAI();
            await Task.Delay(500, ct);

            if (room.ProceedButton.IsEnabled) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;

            attempts++;
        }

        NProceedButton proceedButton = room.ProceedButton;
        await WaitHelper.Until(() => proceedButton.IsEnabled || (NMapScreen.Instance?.IsOpen ?? false), ct, TimeSpan.FromSeconds(5), "Proceed button not enabled");
        
        if (proceedButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(proceedButton);
        }
    }
}
