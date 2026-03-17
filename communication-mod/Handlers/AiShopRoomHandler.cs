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
        // Track consecutive no-affordable-items loops to detect stall
        int noAffordableCount = 0;
        const int MaxNoAffordableBeforeProceed = 3;

        while (!ct.IsCancellationRequested)
        {
            if (MainFile.IsGameBusy())
            {
                await Task.Delay(100, ct);
                continue;
            }

            // Check if there are any affordable items in inventory
            var inventoryNode = room.Inventory;
            bool hasAffordable = false;
            if (inventoryNode != null && inventoryNode.Visible)
            {
                var slots = inventoryNode.GetAllSlots();
                foreach (var slot in slots)
                {
                    if (slot.Entry != null && slot.Entry.IsStocked && slot.Entry.EnoughGold)
                    {
                        hasAffordable = true;
                        break;
                    }
                }
            }

            if (!hasAffordable)
            {
                noAffordableCount++;
                MainFile.Logger.Info($"[AiSlayer] No affordable items in shop (count: {noAffordableCount}/{MaxNoAffordableBeforeProceed})");
                if (noAffordableCount >= MaxNoAffordableBeforeProceed)
                {
                    MainFile.Logger.Info("[AiSlayer] No affordable items - forcing shop exit");
                    break;
                }
            }
            else
            {
                noAffordableCount = 0; // Reset if items become available

                // AI decides what to buy or when to leave
                await MainFile.Instance.StepAI();
                await Task.Delay(200, ct);
            }

            // Exit loop if proceed button is ready or map is open
            if (room.ProceedButton.IsEnabled) break;
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;

            await Task.Delay(100, ct);
        }

        // Close inventory if open before proceeding
        if (room.Inventory != null && room.Inventory.Visible)
        {
            MainFile.Logger.Info("[AiSlayer] Closing inventory before proceed");
            var backButton = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.CommonUi.NBackButton>((Node)(object)room.Inventory);
            if (backButton != null)
            {
                await UiHelper.Click(backButton);
                await Task.Delay(400, ct);
            }
        }

        // Wait for proceed button to become enabled (with timeout)
        int waitCount = 0;
        while (!room.ProceedButton.IsEnabled && waitCount < 20 && !ct.IsCancellationRequested)
        {
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
            await Task.Delay(200, ct);
            waitCount++;
        }

        if (room.ProceedButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiSlayer] Clicking proceed");
            await UiHelper.Click(room.ProceedButton);
        }
        else if (NMapScreen.Instance == null || !NMapScreen.Instance.IsOpen)
        {
            MainFile.Logger.Info("[AiSlayer] Proceed button not enabled, trying ForceClick anyway");
            room.ProceedButton.Call("ForceClick");
        }
    }
}
