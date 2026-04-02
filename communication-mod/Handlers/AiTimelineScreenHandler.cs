using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;
using MegaCrit.Sts2.Core.Random;

namespace communication_mod.Handlers;

/// <summary>
/// Handler for the main timeline screen (NTimelineScreen).
/// </summary>
public class AiTimelineScreenHandler : IScreenHandler
{
    public Type ScreenType => typeof(NTimelineScreen);

    public TimeSpan Timeout => TimeSpan.FromMinutes(2);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiTimelineScreenHandler] Timeline screen detected. Handling...");

        NTimelineScreen? screen = NOverlayStack.Instance?.Peek() as NTimelineScreen;
        if (screen == null)
        {
            MainFile.Logger.Error("[AiTimelineScreenHandler] Screen is not NTimelineScreen");
            return;
        }

        // Wait a bit for animations or queued screens to start
        await Task.Delay(500, ct); // Reduced from 2000ms

        // NTimelineScreen manages a queue of unlock screens.
        // It will automatically open them one by one.
        // During that time, the NTimelineScreen itself will be on the overlay stack, but
        // any specific NUnlockScreen will be high up on the overlay stack too, and
        // DrainOverlayScreensAsync will pick them up first.

        // If no more screens are queued and the timeline is idle, we return to the Main Menu.
        while (!ct.IsCancellationRequested)
        {
            if (screen.IsScreenQueued())
            {
                MainFile.Logger.Info("[AiTimelineScreenHandler] Unlock screen(s) are queued. Waiting for them to appear and be handled by AiUnlockScreenHandler.");
                // We just wait here; DrainOverlayScreensAsync will see the new screen on top.
                await Task.Delay(250, ct); // Reduced from 1000ms
                continue;
            }

            // No screens are queued; check for the Back button.
            NBackButton? backButton = UiHelper.FindFirst<NBackButton>(screen);
            if (backButton != null && backButton.IsEnabled && backButton.Visible && backButton.IsVisibleInTree())
            {
                MainFile.Logger.Info("[AiTimelineScreenHandler] No more screens queued. Clicking Back button to return to Main Menu.");
                await UiHelper.Click(backButton);
                break;
            }

            // Fallback: If no back button, maybe wait a bit longer or exit the handler and let the slayer repeat.
            MainFile.Logger.Info("[AiTimelineScreenHandler] No more screens queued, but Back button not findable/enabled yet. Waiting...");
            await Task.Delay(500, ct); // Reduced from 2000ms
            
            // Re-check validity
            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree()) break;
        }
        
        MainFile.Logger.Info("[AiTimelineScreenHandler] Timeline screen handling complete.");
    }
}
