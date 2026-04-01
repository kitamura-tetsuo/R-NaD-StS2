using System;
using System.Collections.Generic;
using System.Linq;
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

public class AiShopRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/MerchantRoom";
    public RoomType[] HandledTypes => new[] { RoomType.Shop };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for shop room");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        NMerchantRoom room = null;
        await WaitHelper.Until(() => {
            room = UiHelper.FindFirst<NMerchantRoom>(root);
            return room != null;
        }, ct, TimeSpan.FromSeconds(30), "Merchant room not found");

        MainFile.Logger.Info("[AiSlayer] Opening merchant inventory");
        room.OpenInventory();
        await Task.Delay(500, ct);

        MainFile.Logger.Info("[AiSlayer] Handling shopping via AI");
        int attempts = 0;
        int maxAttempts = 50;

        while (attempts < maxAttempts)
        {
            ct.ThrowIfCancellationRequested();
            

            // Check if there are any affordable items in inventory
            bool hasAffordable = false;
            if (room.Inventory != null && room.Inventory.Visible)
            {
                var slots = room.Inventory.GetAllSlots();
                hasAffordable = slots.Any(slot => slot.Entry != null && slot.Entry.IsStocked && slot.Entry.EnoughGold);
            }

            if (!hasAffordable)
            {
                MainFile.Logger.Info("[AiSlayer] No more affordable items - finishing shopping");
                break;
            }

            // AI decides what to buy
            await MainFile.Instance.StepAI(MainFile.Instance.ExecuteShopAction);
            await Task.Delay(500, ct);

            // Exit loop if map is open
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;

            attempts++;
        }

        // Close inventory
        if (room.Inventory != null && room.Inventory.Visible)
        {
            MainFile.Logger.Info("[AiSlayer] Closing inventory");
            var backButton = UiHelper.FindFirst<NBackButton>((Node)(object)room);
            if (backButton != null)
            {
                await UiHelper.Click(backButton);
                await Task.Delay(300, ct);
            }
        }

        // Click Proceed
        NProceedButton proceedButton = room.ProceedButton;
        if (proceedButton != null)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(proceedButton);
        }
    }
}
