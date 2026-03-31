using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;
using MegaCrit.Sts2.Core.Random;

namespace communication_mod.Handlers;

/// <summary>
/// General handler for all unlock screens inheriting from NUnlockScreen.
/// </summary>
public class AiUnlockScreenHandler : IScreenHandler
{
    public Type ScreenType { get; }

    public AiUnlockScreenHandler(Type screenType)
    {
        ScreenType = screenType;
    }

    public TimeSpan Timeout => TimeSpan.FromMinutes(2);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info($"[AiUnlockScreenHandler] Handling {ScreenType.Name}...");

        NUnlockScreen? screen = NOverlayStack.Instance?.Peek() as NUnlockScreen;
        if (screen == null)
        {
            MainFile.Logger.Error($"[AiUnlockScreenHandler] Screen is not {ScreenType.Name}");
            return;
        }

        // Wait for the Confirm button
        // Based on NUnlockScreen decompiler, the button is named "ConfirmButton" and is an NUnlockConfirmButton.
        NUnlockConfirmButton? confirmButton = UiHelper.FindFirst<NUnlockConfirmButton>(screen);
        if (confirmButton == null)
        {
            MainFile.Logger.Error("[AiUnlockScreenHandler] ConfirmButton not found on unlock screen");
            return;
        }

        MainFile.Logger.Info("[AiUnlockScreenHandler] Waiting for Confirm button to be enabled...");
        await WaitHelper.Until(() => confirmButton.IsEnabled && confirmButton.Visible && confirmButton.IsVisibleInTree(), ct, TimeSpan.FromSeconds(30), "Confirm button did not become enabled");

        MainFile.Logger.Info("[AiUnlockScreenHandler] Clicking Confirm button.");
        await UiHelper.Click(confirmButton);

        // Wait for the screen to close
        await WaitHelper.Until(() => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(), ct, TimeSpan.FromSeconds(30), "Unlock screen did not close");
        MainFile.Logger.Info($"[AiUnlockScreenHandler] {ScreenType.Name} closed.");
    }
}
