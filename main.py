import sys
import os
import time
import signal
import threading

# Add subdirectories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "telemetry"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emotions"))

from aggregator import TelemetryAggregator
from daemon import EmotionDaemon

def print_banner():
    banner = """
    +--------------------------------------------------------------+
    |                                                              |
    |      EMOTIONAL COMPUTER: FULL PIPELINE ACTIVE                |
    |                                                              |
    +--------------------------------------------------------------+
    """
    print(banner)

def main():
    print_banner()
    
    db_path = "telemetry_aggregated.db"
    
    # 1. Initialize the Emotion Daemon (Analysis & Labeling)
    # This runs in its own background thread
    print(f"[*] Starting Emotion Daemon (watching {db_path})...")
    emotion_daemon = EmotionDaemon(db_path=db_path, retrain_interval=5)
    emotion_daemon.start()
    
    # 2. Initialize and Run the Telemetry Aggregator (Collection)
    # This is a blocking call that runs the main loop
    print(f"[*] Starting Telemetry Aggregator (updating every 30s)...")
    aggregator = TelemetryAggregator(interval_seconds=30, db_file=db_path)
    
    def signal_handler(sig, frame):
        print("\n[!] Shutdown signal received. Cleaning up...")
        aggregator.daemon.stop()
        aggregator.log_counter.stop()
        emotion_daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("[+] System is LIVE. Press Ctrl+C to exit.\n")
    
    # The aggregator run loop is blocking and will keep the script alive
    try:
        aggregator.run()
    except Exception as e:
        print(f"\n[ERROR] System failure: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()
