using System;
using System.IO;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Text.Json;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Runs;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Context;
using Logger = MegaCrit.Sts2.Core.Logging.Logger;
using HarmonyLib;

namespace recorder_mod
{
    public class DataCollector
    {
        private static readonly Logger Logger = new Logger("RecorderData", LogType.Generic);
        public static DataCollector? Instance { get; private set; }
        private string? _logFilePath;
        private ActionExecutor? _lastExecutor;
        public bool IsInitialized { get; private set; } = false;
        private bool _hasUploaded = false;
        private static readonly System.Net.Http.HttpClient _httpClient = new System.Net.Http.HttpClient();
        private const string DefaultWebhookUrl = "https://discord.com/api/webhooks/1491584705391231156/Q4_k5o5E9NS4Vq2kY3fNZ5JtwSeHESFCc0nTEb7aEp8DO_uMVHcunX3MUKg_l803tF6d";

        public void Initialize()
        {
            if (IsInitialized) return;
            Instance = this;
            
            IsInitialized = true;
            MainFile.Logger.Info("[RecorderData] DataCollector initializing (string-based reflection)...");
            
            // Diagnostic: Search for RunManager by name to avoid TypeLoadException
            try {
                Type? rmType = null;
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies()) {
                    if (assembly.FullName.Contains("sts2")) {
                        rmType = assembly.GetType("MegaCrit.Sts2.Core.Runs.RunManager");
                        if (rmType != null) {
                            MainFile.Logger.Info($"[RecorderData] [Recorder] Found RunManager Type in Assembly {assembly.FullName}");
                            break;
                        }
                    }
                }

                if (rmType != null) {
                    var methods = rmType.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy);
                    foreach (var m in methods) {
                        string n = m.Name.ToLower();
                        if (n.Contains("run") || n.Contains("start") || n.Contains("event") || n.Contains("reward") || n.Contains("choice")) {
                            MainFile.Logger.Info($"[RecorderData] [Recorder] Found RunManager Method: {m.Name}");
                        }
                    }

                    // Subscribe to RunStarted via reflection to be safe
                    var instanceProp = rmType.GetProperty("Instance", BindingFlags.Static | BindingFlags.Public);
                    var rmInstance = instanceProp?.GetValue(null);
                    if (rmInstance != null) {
                        var runStartedEvent = rmType.GetEvent("RunStarted");
                        if (runStartedEvent != null) {
                            var handler = typeof(DataCollector).GetMethod(nameof(OnRunStartedReflection), BindingFlags.Instance | BindingFlags.NonPublic);
                            if (handler != null) {
                                var delegateHandler = Delegate.CreateDelegate(runStartedEvent.EventHandlerType, this, handler);
                                runStartedEvent.AddEventHandler(rmInstance, delegateHandler);
                                MainFile.Logger.Info("[RecorderData] Subscribed to RunManager.RunStarted via reflection.");
                            }
                        }
                        
                        // Also subscribe to Executor
                        SubscribeToExecutorReflection(rmInstance);
                    }
                }
                
                // Install explicit event patch
                InstallEventPatch();

            } catch (Exception e) {
                MainFile.Logger.Error($"[RecorderData] Reflection Error: {e.Message}\n{e.StackTrace}");
            }

            MainFile.Logger.Info("[RecorderData] DataCollector initialized.");
        }

        private static void InstallEventPatch()
        {
            try {
                var harmony = new Harmony("com.kitamura-tetsuo.recorder-mod.events");
                Type? syncType = null;
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies()) {
                    if (assembly.FullName.Contains("sts2")) {
                        syncType = assembly.GetType("MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer");
                        if (syncType != null) break;
                    }
                }

                if (syncType != null) {
                    var original = syncType.GetMethod("ChooseLocalOption", BindingFlags.Instance | BindingFlags.Public);
                    var prefix = typeof(DataCollector).GetMethod(nameof(OnEventOptionChosenReflection), BindingFlags.Static | BindingFlags.NonPublic);
                    
                    if (original != null && prefix != null) {
                        harmony.Patch(original, prefix: new HarmonyMethod(prefix));
                        MainFile.Logger.Info("[RecorderData] Successfully patched EventSynchronizer.ChooseLocalOption via Reflection.");
                    }
                }
            }
            catch (Exception ex) {
                MainFile.Logger.Error($"[RecorderData] Failed to patch EventSynchronizer: {ex.Message}");
            }
        }

        private static void OnEventOptionChosenReflection(object __instance, int index)
        {
            if (!MainFile.Instance.CollectionMode || Instance == null) return;
            try {
                var getLocalEventMethod = __instance.GetType().GetMethod("GetLocalEvent", BindingFlags.Instance | BindingFlags.Public);
                var localEvent = getLocalEventMethod?.Invoke(__instance, null);
                if (localEvent == null) return;

                string actionType = "EventChoice";
                string actionId = index.ToString();

                string stateJson = MainFile.Instance.GetJsonState();
                if (stateJson == null) return;

                var logEntry = new Dictionary<string, object>
                {
                    ["timestamp"] = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                    ["action"] = $"Event Choice {index}",
                    ["action_type"] = actionType,
                    ["action_id"] = actionId,
                    ["state"] = JsonDocument.Parse(stateJson).RootElement
                };

                // Add specifics
                var optionsProp = localEvent.GetType().GetProperty("CurrentOptions", BindingFlags.Instance | BindingFlags.Public);
                var optionsList = optionsProp?.GetValue(localEvent) as System.Collections.IList;
                if (optionsList != null && index >= 0 && index < optionsList.Count)
                {
                    var opt = optionsList[index];
                    
                    var textKeyProp = opt?.GetType().GetProperty("TextKey", BindingFlags.Instance | BindingFlags.Public);
                    var textKeyObj = textKeyProp?.GetValue(opt);
                    var keyProp = textKeyObj?.GetType().GetProperty("Key", BindingFlags.Instance | BindingFlags.Public);
                    string? keyStr = keyProp?.GetValue(textKeyObj) as string;

                    var idProp = localEvent.GetType().GetProperty("Id", BindingFlags.Instance | BindingFlags.Public);
                    var idObj = idProp?.GetValue(localEvent);
                    var entryProp = idObj?.GetType().GetProperty("Entry", BindingFlags.Instance | BindingFlags.Public);
                    string? entryStr = entryProp?.GetValue(idObj) as string;

                    if (entryStr != null) logEntry["event_name"] = entryStr;
                    if (keyStr != null) logEntry["choice_text_key"] = keyStr;
                }

                string line = JsonSerializer.Serialize(logEntry, MainFile.JsonOptions);
                string path = Instance.GetLogPath();
                File.AppendAllText(path, line + "\n");
                MainFile.Logger.Info($"[RecorderData] Logged event choice via Reflection: {index}");
            }
            catch (Exception e) {
                MainFile.Logger.Error($"[RecorderData] Event Logging Reflection Error: {e.Message}");
            }
        }

        private void SubscribeToExecutorReflection(object rmInstance)
        {
            try {
                var executorProp = rmInstance.GetType().GetProperty("ActionExecutor", BindingFlags.Instance | BindingFlags.Public);
                var executor = executorProp?.GetValue(rmInstance) as ActionExecutor;
                if (executor != null && executor != _lastExecutor)
                {
                    if (_lastExecutor != null) {
                        _lastExecutor.BeforeActionExecuted -= OnBeforeActionExecuted;
                    }
                    executor.BeforeActionExecuted -= OnBeforeActionExecuted;
                    executor.BeforeActionExecuted += OnBeforeActionExecuted;
                    _lastExecutor = executor;
                    MainFile.Logger.Info($"[RecorderData] Subscribed to ActionExecutor events (Hash: {executor.GetHashCode()}).");
                }
            } catch {}
        }

        private void OnRunStartedReflection(object state)
        {
            // The type is RunState but we use object to avoid compile-time dependency issues
            MainFile.Logger.Info("[RecorderMod] OnRunStartedReflection triggered.");
            
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string dir = Path.Combine(OS.GetUserDataDir(), "logs");
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            _logFilePath = Path.Combine(dir, $"human_play_{timestamp}.jsonl");
            MainFile.Logger.Info($"[RecorderMod] New run started. Logging to {_logFilePath}");
            
            // Re-subscribe to executor as it might change
            var rmInstance = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (rmInstance != null) SubscribeToExecutorReflection(rmInstance);

            _hasUploaded = false;
            MainFile.Logger.Info("[RecorderMod] Reset upload status for new run.");
        }

        private void OnBeforeActionExecuted(GameAction action)
        {
            RecordAction(action);
        }

        private HashSet<string> _recordedActionIds = new HashSet<string>();

        public void RecordAction(GameAction? action)
        {
            if (!MainFile.Instance.CollectionMode) return;
            
            string actionType = action?.GetType().Name ?? "UIInteraction";
            string actionId = action?.Id.ToString() ?? "0";
            
            // Check if already recorded
            string actionKey = $"{actionId}_{actionType}";
            if (action != null && _recordedActionIds.Contains(actionKey)) return;
            if (action != null) _recordedActionIds.Add(actionKey);
            if (_recordedActionIds.Count > 1000) _recordedActionIds.Clear();

            MainFile.Logger.Info($"[RecorderData] DEBUG: RecordAction called for {actionType}");

            try 
            {
                string stateJson = MainFile.Instance.GetJsonState();
                if (stateJson == null) return;

                var logEntry = new Dictionary<string, object>
                {
                    ["timestamp"] = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                    ["action"] = action?.ToString() ?? "UI Click",
                    ["action_type"] = actionType,
                    ["action_id"] = actionId,
                    ["state"] = JsonDocument.Parse(stateJson).RootElement
                };

                if (action is PlayCardAction pca) {
                     logEntry["card_id"] = pca.CardModelId.Entry;
                     logEntry["target_id"] = pca.TargetId;
                }

                string line = System.Text.Json.JsonSerializer.Serialize(logEntry, MainFile.JsonOptions);
                File.AppendAllText(GetLogPath(), line + "\n");
                
                MainFile.Logger.Info($"[RecorderData] Logged action: {actionType}");
            }
            catch (Exception e)
            {
                MainFile.Logger.Error($"[RecorderData] RecordAction Error: {e.Message}");
            }
        }

        public string GetLogPath()
        {
            if (_logFilePath != null) return _logFilePath;
            
            string dir = Path.Combine(OS.GetUserDataDir(), "logs");
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            _logFilePath = Path.Combine(dir, "human_play_default.jsonl");
            return _logFilePath;
        }
        
        public void Update()
        {
            var rmInstance = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            if (rmInstance != null) {
                SubscribeToExecutorReflection(rmInstance);
                
                // Check for Game Over / Victory
                if (!_hasUploaded && _logFilePath != null && File.Exists(_logFilePath)) {
                    var runState = rmInstance.DebugOnlyGetState();
                    bool isGameOver = false;
                    
                    if (runState != null && runState.IsGameOver) {
                        isGameOver = true;
                        MainFile.Logger.Info("[RecorderData] RunState.IsGameOver detected.");
                    } else {
                        // Check for GameOverScreen overlay as a fallback
                        var topOverlay = MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NOverlayStack.Instance?.Peek();
                        if (topOverlay != null && topOverlay.GetType().FullName.Contains("GameOverScreen")) {
                            isGameOver = true;
                            MainFile.Logger.Info("[RecorderData] GameOverScreen detected.");
                        }
                    }

                    if (isGameOver) {
                        _hasUploaded = true;
                        // Fire and forget upload
                        _ = UploadToDiscordAsync(_logFilePath);
                    }
                }
            }
        }

        private async Task UploadToDiscordAsync(string path)
        {
            try {
                string? webhookUrl = System.Environment.GetEnvironmentVariable("DISCORD_WEBHOOK_URL");
                if (string.IsNullOrEmpty(webhookUrl)) {
                    webhookUrl = DefaultWebhookUrl;
                }

                MainFile.Logger.Info($"[RecorderData] Uploading to Discord: {path}");
                
                using (var content = new MultipartFormDataContent()) {
                    var fileBytes = File.ReadAllBytes(path);
                    var fileContent = new ByteArrayContent(fileBytes);
                    fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse("application/octet-stream");
                    content.Add(fileContent, "file", Path.GetFileName(path));
                    content.Add(new StringContent($"Game Over / Clear Trajectory: {Path.GetFileName(path)}"), "content");

                    var response = await _httpClient.PostAsync(webhookUrl, content);
                    if (response.IsSuccessStatusCode) {
                        MainFile.Logger.Info("[RecorderData] Successfully uploaded to Discord.");
                    } else {
                        string errorStr = await response.Content.ReadAsStringAsync();
                        MainFile.Logger.Error($"[RecorderData] Discord upload failed: {response.StatusCode} - {errorStr}");
                    }
                }
            } catch (Exception e) {
                MainFile.Logger.Error($"[RecorderData] Discord upload error: {e.Message}");
            }
        }
    }
}
