import os
import sys
import time
import threading

BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
LOG_DIR = os.path.join(BRIDGE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rnad_bridge.log")
MAX_LOG_SIZE = 10 * 1024 * 1024 # 10MB
BACKUP_COUNT = 3

class Logger:
    def __init__(self, original, filename):
        self.original = original
        self.filename = filename
        self.file_handle = open(filename, "a", buffering=1)
        self.lock = threading.Lock()

    def _rotate_logs(self):
        self.file_handle.close()
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = f"{self.filename}.{i}"
            dst = f"{self.filename}.{i+1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
            
        self.file_handle = open(self.filename, "w", buffering=1)

    def write(self, message):
        if self.original:
            self.original.write(message)
        
        with self.lock:
            try:
                if os.path.exists(self.filename) and os.path.getsize(self.filename) > MAX_LOG_SIZE:
                    self._rotate_logs()
                self.file_handle.write(message)
                self.file_handle.flush()
            except Exception as e:
                # Fallback to original if file write fails, avoid infinite loops
                if self.original:
                    self.original.write(f"\n[Logger Error] {e}\n")

    def flush(self):
        if self.original:
            self.original.flush()
        with self.lock:
            try:
                self.file_handle.flush()
            except:
                pass

def setup_redirection():
    if not hasattr(sys.stdout, "is_rnad_logger"):
        sys.stdout = Logger(sys.stdout, LOG_FILE)
        sys.stderr = Logger(sys.stderr, LOG_FILE)
        sys.stdout.is_rnad_logger = True

# Dedicated decision logger
DECISION_LOG_FILE = os.path.join(LOG_DIR, "rnad_decisions.log")
decision_logger = Logger(None, DECISION_LOG_FILE)

def log_decision(msg):
    decision_logger.write(f"{time.ctime()}: {msg}\n")

def log(msg):
    print(f"[Python][SM:{id(sys.modules)}][P:{os.getpid()}] {msg}")
