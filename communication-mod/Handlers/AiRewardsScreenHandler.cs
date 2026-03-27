using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;

namespace communication_mod.Handlers;

public class AiRewardsScreenHandler : IScreenHandler
{
    public Type ScreenType => typeof(NRewardsScreen);
    public TimeSpan Timeout => TimeSpan.FromMinutes(1);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiRewardsScreenHandler] Started");
        
        while (!ct.IsCancellationRequested)
        {
            var stack = NOverlayStack.Instance;
            if (stack == null || !GodotObject.IsInstanceValid(stack) || stack.ScreenCount == 0) break;
            
            var peek = stack.Peek();
            if (peek == null || !GodotObject.IsInstanceValid(peek as GodotObject) || peek.GetType() != typeof(NRewardsScreen)) break;

            NRewardsScreen screen = (NRewardsScreen)peek;

            // Find all reward buttons
            var buttons = UiHelper.FindAll<NRewardButton>((Node)(object)screen)
                .Where(b => b.IsEnabled && b.Visible)
                .ToList();

            if (buttons.Count == 0)
            {
                // No more reward buttons, try to find PROCEED
                var proceedBtn = UiHelper.FindFirst<NProceedButton>((Node)(object)screen);
                if (proceedBtn != null && proceedBtn.IsEnabled && proceedBtn.Visible)
                {
                    MainFile.Logger.Info("[AiRewardsScreenHandler] Clicking proceed");
                    await UiHelper.Click(proceedBtn);
                    await Task.Delay(500, ct);
                }
                else
                {
                    MainFile.Logger.Info("[AiRewardsScreenHandler] No buttons left and no Proceed button found/enabled.");
                    break;
                }
                continue;
            }

            // Priority 1: Gold and Relics (Automatic)
            var autoReward = buttons.FirstOrDefault(b => b.Reward is GoldReward || b.Reward is RelicReward);
            if (autoReward != null)
            {
                MainFile.Logger.Info($"[AiRewardsScreenHandler] Automatically picking {autoReward.Reward.GetType().Name}: {autoReward.Reward.Description.GetRawText()}");
                await UiHelper.Click(autoReward);
                await Task.Delay(500, ct);
                continue;
            }

            // Priority 2: Card Rewards (Auto-proceed to selection)
            var cardReward = buttons.FirstOrDefault(b => b.Reward is CardReward || b.Reward is SpecialCardReward);
            if (cardReward != null)
            {
                MainFile.Logger.Info($"[AiRewardsScreenHandler] Automatically proceeding to card reward selection");
                await UiHelper.Click(cardReward);
                await Task.Delay(800, ct);
                
                // Return so DrainOverlayScreensAsync can handle the new screen (NCardRewardSelectionScreen)
                return;
            }

            // If only rewards left are things we don't automate (e.g. Potions), let the AI decide.
            MainFile.Logger.Info("[AiRewardsScreenHandler] No more auto-rewards, stepping AI for remaining items");
            await MainFile.Instance.StepAI(MainFile.Instance.ExecuteUniversalAction);
            await Task.Delay(500, ct);
            
            // Re-check after step
            if (NOverlayStack.Instance?.Peek() != screen) break;
        }
        
        MainFile.Logger.Info("[AiRewardsScreenHandler] Finished");
    }
}
