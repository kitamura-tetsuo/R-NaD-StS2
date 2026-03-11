using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

namespace communication_mod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    private const string
        ModId = "communication-mod"; //At the moment, this is used only for the Logger and harmony names.

    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

        public static Node? AiBridge { get; private set; }

        public static void Initialize()
        {
            Harmony harmony = new(ModId);

            harmony.PatchAll();

            try
            {
                // Manually load the GDExtension from the PCK
                string extensionPath = "res://ai_bridge.gdextension";
                Logger.Info($"Attempting to load GDExtension: {extensionPath}");
                
                var status = GDExtensionManager.LoadExtension(extensionPath);
                Logger.Info($"GDExtension load status: {status}");

                AiBridge = (Node)Godot.ClassDB.Instantiate("AiBridge");
                AiBridge.Name = "MyAiBridge";

            // Access the SceneTree and add the node to the root viewport
            // CallDeferred is used to ensure it is added safely if the tree is currently being modified.
            SceneTree tree = (SceneTree)Engine.GetMainLoop();
            tree.Root.CallDeferred("add_child", AiBridge);

            Logger.Info("AiBridge node successfully instantiated and attached to the SceneTree.");

            // Test the bridge by passing a dummy state
            Variant result = AiBridge!.Call("predict_action", "{\"test\": \"state_from_cs\"}");
            Logger.Info($"Python returned: {result}");
        }
        catch (System.Exception ex)
        {
            Logger.Error($"Failed to instantiate or attach AiBridge node: {ex.Message}");
        }
    }
}