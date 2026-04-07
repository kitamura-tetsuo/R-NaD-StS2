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
        bool hpChecked = false;
        
        while (!ct.IsCancellationRequested)
        {
            var stack = NOverlayStack.Instance;
            if (stack == null || !GodotObject.IsInstanceValid(stack) || stack.ScreenCount == 0) break;
            
            var peek = stack.Peek();
            if (peek == null || !GodotObject.IsInstanceValid(peek as GodotObject) || peek.GetType() != typeof(NRewardsScreen)) break;

            NRewardsScreen screen = (NRewardsScreen)peek;
            
            // HP loss check for combat retry
            if (AiSlayer.IsActive && !hpChecked)
            {
                var slayer = MainFile.Instance.GetAiSlayer();
                var runState = MegaCrit.Sts2.Core.Runs.RunManager.Instance?.DebugOnlyGetState();
                var player = (MegaCrit.Sts2.Core.Entities.Players.Player)MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
                
                if (runState != null && player != null && slayer.HpBeforeCombat > 0)
                {
                    hpChecked = true;
                    int currentHp = (int)player.Creature.CurrentHp;
                    int hpLoss = slayer.HpBeforeCombat - currentHp;
                    int floorThreshold = runState.TotalFloor;
                    
                    // Record trial HP loss in history via Rust bridge (with victory flag)
                    var hpInfo = new Godot.Collections.Dictionary {
                        ["hp_loss"] = hpLoss,
                        ["is_victory"] = true
                    };
                    await MainFile.Instance.CallBridgeSafe("record_hp_loss", Json.Stringify(hpInfo));
                    
                    // Check performance against history
                    var infoDict = new Godot.Collections.Dictionary {
                        ["hp_loss"] = hpLoss
                    };
                    string infoJson = Json.Stringify(infoDict);
                    var perfVariant = await MainFile.Instance.CallBridgeSafe("check_hp_performance", infoJson);

                    bool isTop50 = true;
                    int retryCount = 0;
                    int maxRetries = 3;

                    if (perfVariant.VariantType == Variant.Type.String)
                    {
                        var json = new Json();
                        if (json.Parse(perfVariant.AsString()) == Error.Ok)
                        {
                            var data = json.Data.AsGodotDictionary();
                            isTop50 = data.ContainsKey("is_top_50") ? data["is_top_50"].AsBool() : true;
                            retryCount = data.ContainsKey("retry_count") ? (int)data["retry_count"].AsInt64() : 0;
                            maxRetries = data.ContainsKey("max_retries") ? (int)data["max_retries"].AsInt64() : 3;
                        }
                    }

                    bool shouldRetry = !isTop50;
                    string retryReason = isTop50 ? "" : $"HP loss threshold not met (Attempt {retryCount + 1}/{maxRetries})";

                    if (retryCount >= maxRetries)
                    {
                        shouldRetry = false;
                        MainFile.Logger.Info($"[AiRewardsScreenHandler] Reached max retries ({maxRetries}). Proceeding to next floor without backtracking.");
                    }
                    else if (!shouldRetry && retryCount > 0)
                    {
                        MainFile.Logger.Info($"[AiRewardsScreenHandler] HP loss is within median/Top 50%. Proceeding to next floor.");
                    }
                    
                    if (shouldRetry)
                    {
                        MainFile.Logger.Info($"[AiRewardsScreenHandler] {retryReason}. TRIGGERING RETRY.");
                        
                        await Task.Delay(180, ct);
                        
                        var restoreRes = await MainFile.Instance.CallBridgeSafe("trigger_restore");
                        if (restoreRes.VariantType == Variant.Type.Bool && (bool)restoreRes)
                        {
                            MainFile.Logger.Info("[AiRewardsScreenHandler] Restore successful. Performing Soft Reset (Return to Main Menu).");
                            
                            var dict = new Godot.Collections.Dictionary { ["action"] = "return_to_main_menu" };
                            await MainFile.Instance.ExecuteUniversalAction(dict);
                            return;
                        }
                        else
                        {
                            MainFile.Logger.Error("[AiRewardsScreenHandler] Restore failed or backups exhausted. Proceeding with current rewards.");
                        }
                    }
                }
            }

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
