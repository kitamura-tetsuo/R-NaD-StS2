using System;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;

namespace communication_mod.Handlers;

public class AiScreenHandler : IScreenHandler
{
    public Type ScreenType { get; }
    public AiScreenHandler(Type screenType) { ScreenType = screenType; }
    public TimeSpan Timeout => TimeSpan.FromMinutes(5);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        MainFile.Logger.Info($"[AiSlayer] Screen handler started for {ScreenType.Name}");
        while (!ct.IsCancellationRequested)
        {

            var stack = NOverlayStack.Instance;
            if (stack == null || stack.ScreenCount == 0 || stack.Peek().GetType() != ScreenType) break;

            await MainFile.Instance.StepAI();
            await Task.Delay(500, ct);
        }
    }
}
