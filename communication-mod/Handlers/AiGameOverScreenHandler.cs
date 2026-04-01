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
    public Type ScreenType => typeof(NGameOverScreen);

    public TimeSpan Timeout => TimeSpan.FromMinutes(2);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiGameOverScreenHandler] Game Over screen detected. Handling by code...");
        
        // Notify AI of game over state so it can record episode and trigger restart if needed
        // We capture whether the AI returned a valid step (like 'wait' or 'return_to_main_menu')
        bool stepped = await MainFile.Instance.StepAI(async (d) => {
            string action = d.ContainsKey("action") ? d["action"].AsString() : "";
            if (action == "return_to_main_menu") {
                MainFile.Logger.Info("[AiGameOverScreenHandler] AI requested return to main menu (likely after restore).");
            }
        });
        
        NGameOverScreen? screen = NOverlayStack.Instance?.Peek() as NGameOverScreen;
        if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Screen is no longer valid after StepAI.");
            return;
        }

        // 1. Find and click the Continue button
        NGameOverContinueButton? continueButton = UiHelper.FindFirst<NGameOverContinueButton>(screen);
        if (continueButton == null)
        {
            MainFile.Logger.Error("[AiGameOverScreenHandler] Continue button not found. Checking if screen already closed.");
            if (NOverlayStack.Instance?.Peek() != screen) return;
            return;
        }

        MainFile.Logger.Info("[AiGameOverScreenHandler] Waiting for Continue button to be enabled...");
        try {
            await WaitHelper.Until(() => continueButton.IsEnabled, ct, TimeSpan.FromSeconds(20), "Continue button did not become enabled");
        } catch (Exception ex) {
            MainFile.Logger.Warn($"[AiGameOverScreenHandler] Continue button wait failed: {ex.Message}. Attempting to proceed anyway.");
        }
        
        if (GodotObject.IsInstanceValid(continueButton) && continueButton.Visible && continueButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Continue button");
            await UiHelper.Click(continueButton);
        }

        // 2. Wait for the Return to Main Menu button
        MainFile.Logger.Info("[AiGameOverScreenHandler] Waiting for Return to Main Menu button...");
        NReturnToMainMenuButton? mainMenuButton = null;
        try {
            await WaitHelper.Until(() =>
            {
                if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
                {
                    return true;
                }
                mainMenuButton = UiHelper.FindFirst<NReturnToMainMenuButton>(screen);
                return mainMenuButton != null && mainMenuButton.Visible && mainMenuButton.IsEnabled;
            }, ct, TimeSpan.FromSeconds(30), "Main menu button did not become enabled");
        } catch (Exception ex) {
            MainFile.Logger.Warn($"[AiGameOverScreenHandler] Main menu button wait failed: {ex.Message}");
        }

        if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Game over screen closed automatically or earlier than expected.");
            return;
        }

        if (mainMenuButton != null && mainMenuButton.Visible && mainMenuButton.IsEnabled)
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Return to Main Menu button");
            await UiHelper.Click(mainMenuButton);
        }

        await WaitHelper.Until(() => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(), ct, TimeSpan.FromSeconds(15), "Game over screen did not close");
        MainFile.Logger.Info("[AiGameOverScreenHandler] Game Over screen closed.");
    }
}
