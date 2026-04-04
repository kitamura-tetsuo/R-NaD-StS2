using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
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
    private bool _hasValidBackup = false;

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for combat to start...");
        await WaitHelper.Until(() => {
            if (CombatManager.Instance.IsInProgress) return true;
            
            // Handle dialogue hitboxes if they appear (common in Boss/Elite intro)
            var root = ((Node)(object)((SceneTree)Engine.GetMainLoop()).Root);
            var ancientHitbox = UiHelper.FindFirst<MegaCrit.Sts2.Core.Nodes.Events.NAncientDialogueHitbox>(root);
            if (ancientHitbox != null && ancientHitbox.IsVisibleInTree() && ancientHitbox.IsEnabled)
            {
                MainFile.Logger.Info("[AiSlayer] Clicking ancient dialogue hitbox to start combat.");
                ancientHitbox.Call("ForceClick");
            }
            
            return false;
        }, ct, TimeSpan.FromSeconds(5), "Event room not found"); // Reduced from 30s to 5s
        
        int turnCount = 0;
        _hasValidBackup = false;
        
        // Record HP at the start of combat for retry thresholding
        var slayer = MainFile.Instance.GetAiSlayer();
        var runState = MegaCrit.Sts2.Core.Runs.RunManager.Instance?.DebugOnlyGetState();
        var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
        if (player != null && slayer != null)
        {
            slayer.HpBeforeCombat = (int)player.Creature.CurrentHp;
            MainFile.Logger.Info($"[AiCombatRoomHandler] Recorded HP before combat: {slayer.HpBeforeCombat}");
        }

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
                ct, TimeSpan.FromSeconds(5), "Play phase not started"); // Reduced from 30s to 5s
            
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

                var localPlayer = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState());
                bool isDead = localPlayer != null && localPlayer.Creature != null && localPlayer.Creature.CurrentHp <= 0;

                if (isDead || NOverlayStack.Instance?.Peek() is NGameOverScreen)
                {
                    MainFile.Logger.Info("[AiCombatRoomHandler] Player is dead or Game Over screen detected. Exiting combat loop.");
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

            if (!_hasValidBackup)
            {
                MainFile.Logger.Info($"[AiCombatRoomHandler] End of Turn {turnCount}. Requesting save backup...");
                var result = await MainFile.Instance.CallBridgeSafe("trigger_backup");
                if (result.AsBool())
                {
                    MainFile.Logger.Info($"[AiCombatRoomHandler] Combat backup verified successfully on Turn {turnCount}.");
                    _hasValidBackup = true;
                }
            }
        }

        await WaitHelper.Until(() => !CombatManager.Instance.IsInProgress, ct, TimeSpan.FromSeconds(5), "Combat did not end"); // Reduced from 30s to 5s
        MainFile.Logger.Info("[AiSlayer] Combat finished");
    }
}

