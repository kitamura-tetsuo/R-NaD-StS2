using System;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;

using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

namespace communication_mod.Handlers;

public class AiCombatRoomHandler : IRoomHandler
{
    public RoomType[] HandledTypes => new[] { RoomType.Monster, RoomType.Elite, RoomType.Boss };
    public TimeSpan Timeout => TimeSpan.FromMinutes(10);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for combat to start");
        await WaitHelper.Until(() => CombatManager.Instance.IsInProgress, ct, TimeSpan.FromSeconds(30), "Combat not started");
        
        int turnCount = 0;
        while (CombatManager.Instance.IsInProgress && turnCount < 100)
        {
            ct.ThrowIfCancellationRequested();
            turnCount++;

            // Wait for play phase OR if a selection screen/overlay is open
            await WaitHelper.Until(() => 
                CombatManager.Instance.IsPlayPhase || 
                !CombatManager.Instance.IsInProgress ||
                (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) ||
                (NPlayerHand.Instance != null && NPlayerHand.Instance.IsInCardSelection), 
                ct, TimeSpan.FromSeconds(30), "Play phase not started");
            
            if (!CombatManager.Instance.IsInProgress) break;

            AutoSlayer.CurrentWatchdog?.Reset($"Combat turn {turnCount}");
            MainFile.Logger.Info($"[AiSlayer] Turn {turnCount}: Handling play phase/selection via AI");
            
            int actionsInTurn = 0;
            while (CombatManager.Instance.IsInProgress && actionsInTurn < 100)
            {
                ct.ThrowIfCancellationRequested();

                bool isPlayPhase = CombatManager.Instance.IsPlayPhase;
                bool hasOverlay = NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0;
                bool isHandSelecting = NPlayerHand.Instance != null && NPlayerHand.Instance.IsInCardSelection;

                if (NOverlayStack.Instance?.Peek() is NGameOverScreen)
                {
                    MainFile.Logger.Info("[AiCombatRoomHandler] Game Over screen detected. Exiting combat handler.");
                    return;
                }

                if (!isPlayPhase && !hasOverlay && !isHandSelecting)
                {
                    // Not our turn or busy
                    break;
                }


                await MainFile.Instance.StepAI(MainFile.Instance.ExecuteCombatAction);
                await Task.Delay(500, ct);
                actionsInTurn++;
            }
        }

        await WaitHelper.Until(() => !CombatManager.Instance.IsInProgress, ct, TimeSpan.FromSeconds(30), "Combat did not end");
        MainFile.Logger.Info("[AiSlayer] Combat finished");
    }
}

