using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Random;

namespace communication_mod.Handlers;

public class AiGameOverScreenHandler : IScreenHandler
{
    private bool _isClosing = false;

    public Type ScreenType => typeof(NGameOverScreen);

    public TimeSpan Timeout => TimeSpan.FromMinutes(2);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        await MainFile.Instance.StepAI((dict) => Task.CompletedTask);

        try
        {
        if (AiSlayer.IsActive)
        {
            if (_isClosing)
            {
                MainFile.Logger.Info("[AiGameOverScreenHandler] Transition already in progress. Waiting for screen to close...");
                NGameOverScreen? s = NOverlayStack.Instance?.Peek() as NGameOverScreen;
                if (!GodotObject.IsInstanceValid(s) || !s.IsVisibleInTree())
                {
                    _isClosing = false;
                }
                return;
            }

                MainFile.Logger.Info("[AiGameOverScreenHandler] AI active. Checking for restoration/retry capability...");
                
                var slayer = MainFile.Instance.GetAiSlayer();
                if (slayer.HpBeforeCombat > 0)
                {
                    // Record trial HP loss as total HP before combat (since player is at 0 HP)
                    var hpInfo = new Godot.Collections.Dictionary {
                        ["hp_loss"] = slayer.HpBeforeCombat,
                        ["is_victory"] = false
                    };
                    await MainFile.Instance.CallBridgeSafe("record_hp_loss", Json.Stringify(hpInfo));
                }

                // Call bridge to restore save. This flushes the current trajectory as terminal first.
                var restoreRes = await MainFile.Instance.CallBridgeSafe("trigger_restore");
                if (restoreRes.VariantType == Variant.Type.Bool && (bool)restoreRes)
                {
                    MainFile.Logger.Info("[AiGameOverScreenHandler] Restoration successful. Forcing Return to Main Menu for retry.");
                    await Task.Delay(125, ct); // Brief pause to ensure state is settled (Reduced from 500ms, originally 2000ms)

                    var dict = new Godot.Collections.Dictionary { ["action"] = "return_to_main_menu" };
                    await MainFile.Instance.ExecuteUniversalAction(dict);
                    return;
                }
                else
                {
                    MainFile.Logger.Warn("[AiGameOverScreenHandler] Restoration failed or no backups available. Proceeding with regular Game Over flow.");
                }
            }

            // Notify AI of game over state. Execute the returned action (e.g. return_to_main_menu).
            bool stepped = await MainFile.Instance.StepAI(MainFile.Instance.ExecuteGameOverAction);
            
            if (stepped)
            {
                MainFile.Logger.Info("[AiGameOverScreenHandler] AI took an action. Exiting handler.");
                return;
            }

            NGameOverScreen? screen = NOverlayStack.Instance?.Peek() as NGameOverScreen;
            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
            {
                MainFile.Logger.Info("[AiGameOverScreenHandler] Screen is no longer valid after StepAI.");
                return;
            }

            // 1. Wait and click relevant buttons in a unified loop to handle any order/skipping
            MainFile.Logger.Info("[AiGameOverScreenHandler] Polling for Game Over actions (Continue or Return to Main Menu)...");
            
            while (GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree())
            {
                ct.ThrowIfCancellationRequested();

                var continueBtn = UiHelper.FindFirst<NGameOverContinueButton>(screen);
                if (continueBtn != null && continueBtn.Visible && continueBtn.IsEnabled)
                {
                    MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Continue button");
                    await UiHelper.Click(continueBtn);
                    await Task.Delay(250, ct); // Brief wait for state change
                    continue;
                }

                var mainMenuBtn = UiHelper.FindFirst<NReturnToMainMenuButton>(screen);
                if (mainMenuBtn != null && mainMenuBtn.Visible && mainMenuBtn.IsEnabled)
                {
                    MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Return to Main Menu button");
                    await UiHelper.Click(mainMenuBtn);
                    _isClosing = true;
                    // Exit the loop and wait for screen to close
                    break;
                }

                // If neither button is ready, wait slightly and re-poll
                await Task.Delay(100, ct);
                
                // If we've been waiting here too long, throw a timeout (effectively handled by IScreenHandler context usually, 
                // but let's be safe within the local loop if the buttons never become enabled)
                // However, WaitHelper is better for this. Let's use a WaitHelper.Until for the 'state change' condition instead of manual loop if possible.
                // Re-using WaitHelper.Until for the 'something is ready' condition:
                await WaitHelper.Until(() => {
                    if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree()) return true;
                    var c = UiHelper.FindFirst<NGameOverContinueButton>(screen);
                    if (c != null && c.Visible && c.IsEnabled) return true;
                    var m = UiHelper.FindFirst<NReturnToMainMenuButton>(screen);
                    if (m != null && m.Visible && m.IsEnabled) return true;
                    return false;
                }, ct, TimeSpan.FromSeconds(30), "Neither Continue nor Main Menu button appeared");

            }

            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
            {
                MainFile.Logger.Info("[AiGameOverScreenHandler] Game over screen closed automatically or earlier than expected.");
                _isClosing = false;
                return;
            }

            await WaitHelper.Until(() => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(), ct, TimeSpan.FromSeconds(15), "Game over screen did not close"); // Restored from 2s to 15s
            MainFile.Logger.Info("[AiGameOverScreenHandler] Game Over screen closed normally.");
            _isClosing = false;
        }
        catch (Exception ex)
        {
            MainFile.Logger.Error($"[AiGameOverScreenHandler] CRITICAL ERROR during Game Over handling: {ex.Message}");
            MainFile.Logger.Info("[AiGameOverScreenHandler] Triggering forced ReturnToMainMenu fallback.");
            if (MegaCrit.Sts2.Core.Nodes.NGame.Instance != null)
            {
                _isClosing = true;
                MegaCrit.Sts2.Core.Nodes.NGame.Instance.ReturnToMainMenu();
            }
        }
    }
}
