using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using communication_mod.Handlers;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Timeline;

namespace communication_mod;

public class AiSlayer
{
    private readonly Dictionary<RoomType, IRoomHandler> _roomHandlers;
    private readonly Dictionary<Type, IScreenHandler> _screenHandlers;
    private readonly AiMapScreenHandler _mapHandler;

    private CancellationTokenSource? _cts;
    private Rng? _random;
    private Watchdog? _watchdog;
    private string _currentSeed = "";

    public static bool IsActive { get; private set; }

    public AiSlayer()
    {
        _roomHandlers = new Dictionary<RoomType, IRoomHandler>
        {
            [RoomType.Monster] = new AiCombatRoomHandler(),
            [RoomType.Elite] = new AiCombatRoomHandler(),
            [RoomType.Boss] = new AiCombatRoomHandler(),
            [RoomType.Event] = new AiEventRoomHandler(),
            [RoomType.Shop] = new AiShopRoomHandler(),
            [RoomType.Treasure] = new AiTreasureRoomHandler(),
            [RoomType.RestSite] = new AiRestSiteRoomHandler()
        };

        _mapHandler = new AiMapScreenHandler();
        _screenHandlers = new Dictionary<Type, IScreenHandler>
        {
            [typeof(NGameOverScreen)] = new AiGameOverScreenHandler()
        };
    }

    public void Start(string seed)
    {
        if (IsActive)
        {
            MainFile.Logger.Info($"[AiSlayer] Already active. Stopping previous run to start new one with seed: {seed}");
            Stop();
        }
        
        IsActive = true;
        _cts = new CancellationTokenSource();
        _ = RunAsync(seed, _cts.Token);
    }

    public void Stop()
    {
        IsActive = false;
        _cts?.Cancel();
        _cts?.Dispose();
        _cts = null;
    }

    private async Task RunAsync(string seed, CancellationToken ct)
    {
        MainFile.Logger.Info($"[AiSlayer] Starting run with seed: {seed}");
        try
        {
            await PlayRunAsync(seed, ct);
            MainFile.Logger.Info($"[AiSlayer] Run completed: {seed}");
        }
        catch (OperationCanceledException)
        {
            MainFile.Logger.Info($"[AiSlayer] Run cancelled: {seed}");
        }
        catch (Exception ex)
        {
            MainFile.Logger.Error($"[AiSlayer] Run failed: {ex.Message}\n{ex.StackTrace}");
        }
        finally
        {
            IsActive = false;
            _watchdog = null;
        }
    }

    private async Task PlayRunAsync(string seed, CancellationToken ct)
    {
        await WaitHelper.Until(() => NGame.Instance != null, ct, TimeSpan.FromSeconds(30), "Game instance not initialized");
        
        _currentSeed = seed;
        _random = new Rng((uint)seed.GetHashCode());
        _watchdog = new Watchdog();
        
        // Wait for EITHER Main Menu OR Run state to appear (handling race condition with programmatic start)
        MainFile.Logger.Info("[AiSlayer] Stage: Waiting for MainMenu or active Run...");
        await WaitHelper.Until(() => {
            var root = ((Node)(object)((SceneTree)Engine.GetMainLoop()).Root);
            bool hasMenu = root.GetNodeOrNull("Game/RootSceneContainer/MainMenu") != null;
            bool inRun = RunManager.Instance?.DebugOnlyGetState()?.CurrentRoom != null;
            return hasMenu || inRun;
        }, ct, TimeSpan.FromSeconds(15), "Neither MainMenu nor Run state appeared");

        bool runActive = RunManager.Instance?.DebugOnlyGetState()?.CurrentRoom != null;
        if (!runActive)
        {
            MainFile.Logger.Info("[AiSlayer] Stage: PlayMainMenu");
            await PlayMainMenuAsync(ct);
        }
        else
        {
            MainFile.Logger.Info("[AiSlayer] Stage: Skipping PlayMainMenu (Run already active)");
        }

        MainFile.Logger.Info("[AiSlayer] Stage: Wait for Run state");
        await WaitHelper.Until(() => RunManager.Instance?.DebugOnlyGetState() != null, ct, TimeSpan.FromSeconds(30), "Run state not initialized");
        
        MainFile.Logger.Info("[AiSlayer] Stage: Wait for room/intro");
        await WaitHelper.Until(() => {
            try {
                var rm = RunManager.Instance;
                var state = rm?.DebugOnlyGetState();
                var overlayStack = NOverlayStack.Instance;
                
                bool roomReady = state?.CurrentRoom != null && state.CurrentRoom.RoomType != RoomType.Unassigned;
                bool overlayReady = overlayStack != null && overlayStack.Peek() != null;
                
                return roomReady || overlayReady;
            } catch (Exception) {
                return false;
            }
        }, ct, TimeSpan.FromSeconds(30), "Room type not assigned or overlay not present");
 
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            RunState runState = RunManager.Instance?.DebugOnlyGetState();
            if (runState == null || runState.TotalFloor >= 60)
            {
                MainFile.Logger.Info($"[AiSlayer] Loop break: runState is {(runState == null ? "null" : "not null")}, floor={runState?.TotalFloor ?? -1}");
                break;
            }

            int floorAtStart = runState.TotalFloor;

            RoomType roomType = runState.CurrentRoom?.RoomType ?? RoomType.Unassigned;
            _watchdog.Reset($"Entering {roomType} room (Floor {runState.TotalFloor})");
            
            MainFile.Logger.Info($"[AiSlayer] Game Loop: floor={runState.TotalFloor}, room={roomType}, overlay={NOverlayStack.Instance?.Peek()?.GetType().FullName ?? "none"}");
            await HandleRoomAsync(roomType, ct);
            
            // Wait for rewards after combat or event rooms (which might trigger rewards)
            if (roomType == RoomType.Monster || roomType == RoomType.Elite || roomType == RoomType.Boss || roomType == RoomType.Event)
            {
                await WaitForRewardsScreenAsync(ct);
            }
            
            await DrainOverlayScreensAsync(ct);
            
            _watchdog.Reset("Navigating map");
            
            // Re-check floor to see if it already advanced (e.g. via premature AI step during transition)
            int currentFloor = RunManager.Instance?.DebugOnlyGetState()?.TotalFloor ?? floorAtStart;
            
            if (currentFloor <= floorAtStart)
            {
                await _mapHandler.HandleAsync(_random, ct);
            }
            else
            {
                MainFile.Logger.Info($"[AiSlayer] Skipping map navigation because floor already advanced from {floorAtStart} to {currentFloor}.");
            }
            
            await Task.Delay(500, ct);
        }
    }

    private async Task PlayMainMenuAsync(CancellationToken ct)
    {
        MainFile.Logger.Info("[AiSlayer] Playing main menu...");
        Node root = (Node)(object)((SceneTree)Engine.GetMainLoop()).Root;
        Control mainMenu = await WaitHelper.ForNode<Control>(root, "/root/Game/RootSceneContainer/MainMenu", ct, TimeSpan.FromSeconds(30));
        
        // Abandon existing run if present
        NButton abandonBtn = mainMenu.GetNodeOrNull<NButton>(new NodePath("MainMenuTextButtons/AbandonRunButton"));
        if (abandonBtn != null && abandonBtn.Visible)
        {
            MainFile.Logger.Info("[AiSlayer] Abandoning existing run...");
            await UiHelper.Click(abandonBtn);
            await WaitHelper.Until(() => NModalContainer.Instance?.OpenModal != null, ct, TimeSpan.FromSeconds(5), "Confirm modal didn't appear");
            Node modal = (Node)NModalContainer.Instance.OpenModal;
            NButton yesBtn = modal.GetNode<NButton>(new NodePath("VerticalPopup/YesButton"));
            await UiHelper.Click(yesBtn);
            await WaitHelper.Until(() => NModalContainer.Instance.OpenModal == null, ct, TimeSpan.FromSeconds(5), "Confirm modal didn't close");
        }

        // If we have a seed, use the programmatic start instead of clicking through UI
        if (!string.IsNullOrEmpty(_currentSeed))
        {
            MainFile.Logger.Info($"[AiSlayer] Starting seeded run programmatically: {_currentSeed}");
            MainFile.Instance.StartSts2Run(_currentSeed);
            return;
        }

        NButton singlePlayerBtn = mainMenu.GetNode<NButton>(new NodePath("MainMenuTextButtons/SingleplayerButton"));
        await UiHelper.Click(singlePlayerBtn);

        // Wait for submenus
        await WaitHelper.Until(() => {
            var charSelect = mainMenu.GetNodeOrNull<Control>(new NodePath("Submenus/CharacterSelectScreen"));
            var standardBtn = mainMenu.GetNodeOrNull<NButton>(new NodePath("Submenus/SingleplayerSubmenu/StandardButton"));
            return (charSelect != null && charSelect.Visible) || (standardBtn != null && standardBtn.Visible);
        }, ct, TimeSpan.FromSeconds(5), "Submenu didn't appear");

        NButton standardRunBtn = mainMenu.GetNodeOrNull<NButton>(new NodePath("Submenus/SingleplayerSubmenu/StandardButton"));
        if (standardRunBtn != null && standardRunBtn.Visible)
        {
            await UiHelper.Click(standardRunBtn);
        }

        Control charSelectScreen = await WaitHelper.ForNode<Control>(mainMenu, "Submenus/CharacterSelectScreen", ct, TimeSpan.FromSeconds(5));
        
        // Select random character (for simplicity or follow seed)
        Node container = charSelectScreen.GetNode(new NodePath("CharSelectButtons/ButtonContainer"));
        var charButtons = container.GetChildren().OfType<MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect.NCharacterSelectButton>().Where(b => !b.IsLocked).ToList();
        var charBtn = _random.NextItem(charButtons);
        
        MainFile.Logger.Info($"[AiSlayer] Selecting character: {charBtn.Character.Id}");
        charBtn.Select();
        await Task.Delay(100, ct);

        NButton confirmBtn = await WaitHelper.ForNode<NButton>(charSelectScreen, "ConfirmButton", ct, TimeSpan.FromSeconds(5));
        await UiHelper.Click(confirmBtn);
    }

    private async Task HandleRoomAsync(RoomType roomType, CancellationToken ct)
    {
        if (_roomHandlers.TryGetValue(roomType, out var handler))
        {
            await handler.HandleAsync(_random, ct);
        }
        else
        {
            MainFile.Logger.Warn($"[AiSlayer] No handler for room type: {roomType}");
        }
    }

    internal async Task DrainOverlayScreensAsync(CancellationToken ct)
    {
        while (true)
        {
            if (ct.IsCancellationRequested) break;
            
            var instance = NOverlayStack.Instance;
            if (instance == null || !GodotObject.IsInstanceValid(instance) || instance.ScreenCount <= 0) break;

            var currentOverlay = instance.Peek();
            if (currentOverlay == null || !GodotObject.IsInstanceValid(currentOverlay as GodotObject)) break;

            Type type = currentOverlay.GetType();
            _watchdog.Reset($"Handling screen: {type.Name}");
            
            // Check for specialized handler
            if (!_screenHandlers.TryGetValue(type, out var handler))
            {
                // Fallback to generic AI screen handler
                handler = new AiScreenHandler(type);
            }
            await handler.HandleAsync(_random, ct);
            
            await Task.Delay(200, ct);
            
            // Re-check validity after await
            if (instance == null || !GodotObject.IsInstanceValid(instance)) break;

            // If screen is still there after handling, we might be stuck or it expects more input
            if (instance.Peek() == currentOverlay)
            {
                MainFile.Logger.Info($"[AiSlayer] Screen {type.Name} still open after handle.");
                break; 
            }
        }
    }

    private async Task WaitForRewardsScreenAsync(CancellationToken ct)
    {
        await WaitHelper.Until(() => 
            NOverlayStack.Instance?.Peek() is MegaCrit.Sts2.Core.Nodes.Screens.NRewardsScreen || 
            NOverlayStack.Instance?.Peek() is NGameOverScreen ||
            (MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance?.IsOpen ?? false), 
            ct, TimeSpan.FromSeconds(20), "Rewards screen or Game Over did not appear");
    }
}
