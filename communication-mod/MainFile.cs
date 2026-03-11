using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

namespace communication_mod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string
        ModId = "communication_mod"; //At the moment, this is used only for the Logger and harmony names.

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    public static void Initialize()
    {
        Harmony harmony = new(ModId);

        harmony.PatchAll();
    }
}