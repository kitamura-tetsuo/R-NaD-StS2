using System;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;

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

            // Wait for play phase
            await WaitHelper.Until(() => CombatManager.Instance.IsPlayPhase || !CombatManager.Instance.IsInProgress, ct, TimeSpan.FromSeconds(30), "Play phase not started");
            
            if (!CombatManager.Instance.IsInProgress) break;

            MainFile.Logger.Info($"[AiSlayer] Turn {turnCount}: Handling play phase via AI");
            while (CombatManager.Instance.IsPlayPhase && CombatManager.Instance.IsInProgress)
            {
                ct.ThrowIfCancellationRequested();

                if (MainFile.IsGameBusy())
                {
                    await Task.Delay(100, ct);
                    continue;
                }

                await MainFile.Instance.StepAI();
                await Task.Delay(200, ct);
            }
        }

        await WaitHelper.Until(() => !CombatManager.Instance.IsInProgress, ct, TimeSpan.FromSeconds(30), "Combat did not end");
        MainFile.Logger.Info("[AiSlayer] Combat finished");
    }
}
