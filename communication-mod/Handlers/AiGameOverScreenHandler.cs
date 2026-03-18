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
        
        NGameOverScreen? screen = NOverlayStack.Instance?.Peek() as NGameOverScreen;
        if (screen == null)
        {
            MainFile.Logger.Error("[AiGameOverScreenHandler] Screen is not NGameOverScreen");
            return;
        }

        // 1. Find and click the Continue button
        NGameOverContinueButton? continueButton = UiHelper.FindFirst<NGameOverContinueButton>(screen);
        if (continueButton == null)
        {
            MainFile.Logger.Error("[AiGameOverScreenHandler] Continue button not found");
            return;
        }

        MainFile.Logger.Info("[AiGameOverScreenHandler] Waiting for Continue button to be enabled...");
        await WaitHelper.Until(() => continueButton.IsEnabled, ct, TimeSpan.FromSeconds(30), "Continue button did not become enabled");
        
        MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Continue button");
        await UiHelper.Click(continueButton);

        // 2. Wait for the Return to Main Menu button
        MainFile.Logger.Info("[AiGameOverScreenHandler] Waiting for Return to Main Menu button...");
        NReturnToMainMenuButton? mainMenuButton = null;
        await WaitHelper.Until(() =>
        {
            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
            {
                return true;
            }
            mainMenuButton = UiHelper.FindFirst<NReturnToMainMenuButton>(screen);
            return mainMenuButton != null && mainMenuButton.Visible && mainMenuButton.IsEnabled;
        }, ct, TimeSpan.FromSeconds(60), "Main menu button did not become enabled");

        if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Game over screen closed automatically or earlier than expected.");
            return;
        }

        if (mainMenuButton != null)
        {
            MainFile.Logger.Info("[AiGameOverScreenHandler] Clicking Return to Main Menu button");
            await UiHelper.Click(mainMenuButton);
        }

        await WaitHelper.Until(() => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(), ct, TimeSpan.FromSeconds(30), "Game over screen did not close");
        MainFile.Logger.Info("[AiGameOverScreenHandler] Game Over screen closed.");
    }
}
