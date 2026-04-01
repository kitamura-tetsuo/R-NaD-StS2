using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Rooms;

using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace communication_mod.Handlers;

public class AiEventRoomHandler : IRoomHandler
{
    private const string _roomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/EventRoom";
    private const int _maxIterations = 50;

    public RoomType[] HandledTypes => new[] { RoomType.Event };
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Waiting for event room");
        Node eventRoom = await WaitForEventRoom(ct);
        
        if (await WaitForEventOptions(eventRoom, ct))
        {
            MainFile.Logger.Info("[AiSlayer] Event room completed (Custom event handled)");
            return;
        }

        int iterations = 0;
        while (iterations < _maxIterations)
        {
            ct.ThrowIfCancellationRequested();
            
            // Check if combat was triggered or if map opened
            if (CombatManager.Instance.IsInProgress)
            {
                MainFile.Logger.Info("[AiSlayer] Event triggered combat, exiting event handler.");
                break;
            }

            if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree()) break;

            // Check if event is finished and we should proceed
            var runState = RunManager.Instance.DebugOnlyGetState();
            var er = runState?.CurrentRoom as MegaCrit.Sts2.Core.Rooms.EventRoom;
            var ev = er?.LocalMutableEvent;

            if (ev != null && ev.IsFinished)
            {
                MainFile.Logger.Info("[AiSlayer] Event is finished, auto-proceeding.");
                await NEventRoom.Proceed();
                await Task.Delay(500, ct);
                
                // Check if map opened or if room vanished
                if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;
                if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree()) break;
                
                iterations++;
                continue;
            }

            // AI decides which event option to click
            var allButtons = UiHelper.FindAll<NEventOptionButton>(eventRoom);
            var enabledOptions = allButtons.Where(o => o.Option != null && !o.Option.IsLocked && o.IsEnabled && o.Visible).ToList();
            
            if (enabledOptions.Count == 1)
            {
                MainFile.Logger.Info($"[AiSlayer] Auto-selecting single event option: {enabledOptions[0].Option.Title.GetRawText()}");
                await UiHelper.Click(enabledOptions[0]);
            }
            else if (enabledOptions.Count > 0)
            {
                MainFile.Logger.Info($"[AiSlayer] Requesting AI decision for event with {enabledOptions.Count} options.");
                var aiTask = MainFile.Instance.StepAI(MainFile.Instance.ExecuteUniversalAction);
                var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10), ct);
                
                var completedTask = await Task.WhenAny(aiTask, timeoutTask);
                if (completedTask == timeoutTask)
                {
                    MainFile.Logger.Warn("[AiSlayer] AI decision TIMEOUT for event. Picking random option as fallback.");
                    var randomOption = enabledOptions[random.NextInt(enabledOptions.Count)];
                    await UiHelper.Click(randomOption);
                }
                else if (!await aiTask) // StepAI returns false if it didn't call the callback (e.g. error)
                {
                    MainFile.Logger.Warn("[AiSlayer] AI decision FAILED for event. Picking random option as fallback.");
                    var randomOption = enabledOptions[random.NextInt(enabledOptions.Count)];
                    await UiHelper.Click(randomOption);
                }
            }
            else
            {
                // If there are no enabled options, check for a proceed button before falling back to AI
                NProceedButton proceedBtn = UiHelper.FindFirst<NProceedButton>(eventRoom);
                if (proceedBtn == null || !proceedBtn.IsEnabled || !proceedBtn.Visible)
                {
                    // Fallback: search more broadly (some events have proceed buttons as siblings or global)
                    var root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
                    proceedBtn = UiHelper.FindFirst<NProceedButton>(root);
                }

                if (proceedBtn != null && proceedBtn.IsEnabled && proceedBtn.Visible)
                {
                    MainFile.Logger.Info("[AiSlayer] No event options but proceed button found, auto-clicking.");
                    await UiHelper.Click(proceedBtn);
                }
                else
                {
                    MainFile.Logger.Info("[AiSlayer] No event options or proceed button found, requesting AI decision.");
                    // Use Universal action executor to handle overlays like card selection which might be part of the event
                    await MainFile.Instance.StepAI(MainFile.Instance.ExecuteUniversalAction);
                }
            }
            await Task.Delay(500, ct);

            // Exit if map is open
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;

            // Wait for next interaction or room exit
            bool stateChanged = false;
            int waitClicks = 0;
            await WaitHelper.Until(delegate
            {
                waitClicks++;
                // Handle dialogue hitboxes if they appear
                NAncientEventLayout ancientLayout = UiHelper.FindFirst<NAncientEventLayout>(eventRoom);
                if (ancientLayout != null)
                {
                    NButton hitBox = ancientLayout.GetNodeOrNull<NButton>(new NodePath("%DialogueHitbox"));
                    if (hitBox != null && hitBox.Visible && hitBox.IsEnabled)
                    {
                        if (waitClicks % 5 == 0) { // Throttle click-through
                           MainFile.Logger.Info("[AiSlayer] Clicking ancient dialogue hitbox to advance.");
                           hitBox.EmitSignal(NClickableControl.SignalName.Released, new Variant[] { hitBox });
                        }
                    }
                }

                if (CombatManager.Instance.IsInProgress) return true;
                if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) return true;
                if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen)
                {
                    MainFile.Logger.Info("[AiSlayer] Map screen detected, exiting event handler.");
                    return true;
                }
                if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree())
                {
                    stateChanged = true;
                    return true;
                }
                
                // If we're waiting after a click, we wait for either the button we clicked to go away/disable, 
                // OR for its option to change, OR a new button to appear.
                var currentOptions = UiHelper.FindAll<NEventOptionButton>(eventRoom).Where(o => o.Option != null && !o.Option.IsLocked && o.IsEnabled && o.Visible).ToList();
                if (currentOptions.Count != enabledOptions.Count) return true;
                if (currentOptions.Count > 0 && enabledOptions.Count > 0 && currentOptions[0] != enabledOptions[0]) return true;

                // Also check if event finished state changed
                if (ev != null && ev.IsFinished) return true;

                return false;
            }, ct, TimeSpan.FromSeconds(5), "Waiting for next event state");

            // Exit if room gone
            if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree()) break;
            
            // Iteration check to prevent infinite loops in weird edge cases
            iterations++;
        }
        
        MainFile.Logger.Info("[AiSlayer] Event room completed");
    }

    private async Task<Node> WaitForEventRoom(CancellationToken ct)
    {
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        NEventRoom room = null;
        await WaitHelper.Until(() => {
            room = UiHelper.FindFirst<NEventRoom>(root);
            return room != null;
        }, ct, TimeSpan.FromSeconds(30), "Event room not found");
        return room;
    }

    private async Task<bool> WaitForEventOptions(Node eventRoom, CancellationToken ct)
    {
        NAncientEventLayout ancientLayout = UiHelper.FindFirst<NAncientEventLayout>(eventRoom);
        if (ancientLayout != null)
        {
            await HandleAncientEventDialogue(ancientLayout, ct);
            return false;
        }

        NFakeMerchant fakeMerchant = UiHelper.FindFirst<NFakeMerchant>(eventRoom);
        if (fakeMerchant != null)
        {
            MainFile.Logger.Info("[AiSlayer] Detected custom event: FakeMerchant");
            await HandleFakeMerchantEvent(fakeMerchant, ct);
            return true;
        }

        await WaitHelper.Until(() => UiHelper.FindAll<NEventOptionButton>(eventRoom).Count > 0 || CombatManager.Instance.IsInProgress, 
            ct, TimeSpan.FromSeconds(30), "Event options not loaded");
        
        return false;
    }

    private async Task HandleFakeMerchantEvent(NFakeMerchant fakeMerchant, CancellationToken ct)
    {
        NProceedButton proceedButton = null;
        await WaitHelper.Until(delegate
        {
            proceedButton = UiHelper.FindFirst<NProceedButton>(fakeMerchant);
            return proceedButton != null && proceedButton.IsEnabled && proceedButton.Visible;
        }, ct, TimeSpan.FromSeconds(10), "FakeMerchant proceed button not available");
        
        MainFile.Logger.Info("[AiSlayer] Clicking FakeMerchant proceed button");
        await UiHelper.Click(proceedButton);
    }

    private async Task HandleAncientEventDialogue(NAncientEventLayout ancientLayout, CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Detected Ancient event, clicking through dialogue");
        int clicks = 0;
        while (clicks < 50)
        {
            ct.ThrowIfCancellationRequested();
            if (!GodotObject.IsInstanceValid(ancientLayout)) break;

            var options = UiHelper.FindAll<NEventOptionButton>(ancientLayout).Where(b => b.IsEnabled && !b.Option.IsLocked).ToList();
            if (options.Count > 0) break;

            NButton hitBox = ancientLayout.GetNodeOrNull<NButton>(new NodePath("%DialogueHitbox"));
            if (hitBox != null && hitBox.Visible && hitBox.IsEnabled)
            {
                MainFile.Logger.Info($"[AiSlayer] Clicking Ancient dialogue (click {clicks + 1})");
                hitBox.EmitSignal(NClickableControl.SignalName.Released, new Variant[] { hitBox });
                clicks++;
                await Task.Delay(500, ct);
            }
            else
            {
                await Task.Delay(100, ct);
            }
        }
    }
}
