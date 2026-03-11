using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Entities.Creatures;
using System.Linq;

namespace communication_mod
{
    [HarmonyPatch(typeof(CardModel), nameof(CardModel.TryManualPlay))]
    public static class CardSelectionPatch
    {
        public static bool IsAiPlaying = false;

        public static bool Prefix(CardModel __instance, Creature target, ref bool __result)
        {
            if (IsAiPlaying) return true; // Let the AI playback go through normally

            // 1. Serialize game state
            var handCards = __instance.Owner?.PlayerCombatState?.Hand?.Cards;
            var handList = new Godot.Collections.Array();
            if (handCards != null)
            {
                foreach (var card in handCards)
                {
                    handList.Add(card.Id.Entry);
                }
            }

            var energy = __instance.Owner?.PlayerCombatState?.Energy ?? 0;
            var hp = __instance.Owner?.Creature?.CurrentHp ?? 0;

            var stateDict = new Godot.Collections.Dictionary
            {
                { "hp", hp },
                { "energy", energy },
                { "hand", handList },
                { "selected_card", __instance.Id.Entry }
            };

            string stateJson = Json.Stringify(stateDict);

            // 2. Pass to R-NaD Python server
            MainFile.Logger.Info($"[CardSelectionPatch] Intercepted manual play! Sending state to R-NaD: {stateJson}");
            
            if (MainFile.AiBridge == null)
            {
                MainFile.Logger.Error("[CardSelectionPatch] AiBridge is null!");
                return true;
            }

            Variant aiResponse = MainFile.AiBridge.Call("predict_action", stateJson);
            string responseStr = aiResponse.AsString();
            MainFile.Logger.Info($"[CardSelectionPatch] R-NaD response: {responseStr}");

            // 3. Parse action and act
            if (string.IsNullOrEmpty(responseStr))
                return true; // Fallback

            var json = new Json();
            var err = json.Parse(responseStr);
            if (err == Error.Ok)
            {
                var dict = json.Data.AsGodotDictionary();
                if (dict.ContainsKey("action") && dict["action"].AsString() == "play_card")
                {
                    string aiCardIdToPlay = dict["card_id"].AsString();
                    MainFile.Logger.Info($"[CardSelectionPatch] AI chose to play: {aiCardIdToPlay}");
                    
                    var cardToPlay = handCards?.FirstOrDefault(c => c.Id.Entry == aiCardIdToPlay);
                    if (cardToPlay != null)
                    {
                        // Override player's choice and play the AI card!
                        IsAiPlaying = true;
                        MainFile.Logger.Info($"[CardSelectionPatch] Executing AI choice: {aiCardIdToPlay}");
                        __result = cardToPlay.TryManualPlay(target);
                        IsAiPlaying = false;
                        
                        return false; // Prevent the original card from playing!
                    }
                    else
                    {
                        MainFile.Logger.Error($"[CardSelectionPatch] AI chose {aiCardIdToPlay} but it's not in hand!");
                    }
                }
            }
            else
            {
                MainFile.Logger.Error($"[CardSelectionPatch] JSON Parse Error: {err} for {responseStr}");
            }

            // Fallback to normal behavior
            return true;
        }
    }
}
