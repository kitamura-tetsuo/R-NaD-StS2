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
            
            if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree())
            {
                // Check if combat was triggered
                if (CombatManager.Instance.IsInProgress)
                {
                    MainFile.Logger.Info("[AiSlayer] Event triggered combat, delegating to AI step");
                    // We just keep stepping AI until combat is done
                    while (CombatManager.Instance.IsInProgress)
                    {
                        await MainFile.Instance.StepAI(MainFile.Instance.ExecuteCombatAction);
                        await Task.Delay(200, ct);
                    }
                    
                    // Check if event resumes
                    Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
                    Node nodeOrNull = root.GetNodeOrNull(_roomPath);
                    if (nodeOrNull == null) break;
                    
                    eventRoom = nodeOrNull;
                    await Task.Delay(500, ct);
                    if (!UiHelper.FindAll<NEventOptionButton>(eventRoom).Any(o => !o.Option.IsLocked)) break;
                    
                    iterations++;
                    continue;
                }
                break;
            }


            // AI decides which event option to click
            var unlockedOptions = UiHelper.FindAll<NEventOptionButton>(eventRoom).Where(o => !o.Option.IsLocked).ToList();
            if (unlockedOptions.Count == 1)
            {
                MainFile.Logger.Info("[AiSlayer] Auto-selecting single event option");
                await UiHelper.Click(unlockedOptions[0]);
            }
            else
            {
                await MainFile.Instance.StepAI(MainFile.Instance.ExecuteEventAction);
            }
            await Task.Delay(500, ct);

            // Exit if map is open
            if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) break;

            // Wait for next interaction or room exit
            bool roomExit = false;
            await WaitHelper.Until(delegate
            {
                if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) return true;
                if (NMapScreen.Instance != null && NMapScreen.Instance.IsOpen) return true;
                if (!GodotObject.IsInstanceValid(eventRoom) || !eventRoom.IsInsideTree())
                {
                    roomExit = true;
                    return true;
                }
                return UiHelper.FindAll<NEventOptionButton>(eventRoom).Any(o => !o.Option.IsLocked);
            }, ct, TimeSpan.FromSeconds(5), "Waiting for next event state");

            if (roomExit) break;
            
            // If overlay opened (like rewards or deck screen), exit this handler to let AiSlayer handle overlays
            if (NOverlayStack.Instance != null && NOverlayStack.Instance.ScreenCount > 0) break;

            iterations++;
        }
        
        MainFile.Logger.Info("[AiSlayer] Event room completed");
    }

    private async Task<Node> WaitForEventRoom(CancellationToken ct)
    {
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        return await WaitHelper.ForNode<Node>(root, _roomPath, ct, null);
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
