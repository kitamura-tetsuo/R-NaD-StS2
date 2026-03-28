using System;
using System.IO;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Text.Json;
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
        private string _logFilePath;
        private ActionExecutor? _lastExecutor;
        public bool IsInitialized { get; private set; } = false;

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
            } catch (Exception e) {
                MainFile.Logger.Error($"[RecorderData] Reflection Error: {e.Message}\n{e.StackTrace}");
            }

            MainFile.Logger.Info("[RecorderData] DataCollector initialized.");
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

        private string GetLogPath()
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
            if (rmInstance != null) SubscribeToExecutorReflection(rmInstance);
        }
    }

    // No Harmony patches in this version to avoid assembly loading issues during patching
}
