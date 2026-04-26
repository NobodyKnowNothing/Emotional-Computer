import psutil
import time
import threading
from collections import deque

try:
    import GPUtil
except ImportError:
    GPUtil = None

class SystemTelemetryDaemon:
    """
    Background daemon to continuously gather system metrics (CPU, RAM, Disk, Battery, GPU)
    and store them in a rolling window.
    """
    def __init__(self, history_seconds=30):
        self.history_seconds = history_seconds
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=history_seconds)
        self._thread = None
        self._running = False

    def _collect(self):
        """Background task to continuously gather metrics."""
        while self._running:
            # psutil.cpu_percent with interval=1 blocks for 1 second.
            # This calculates accurate CPU usage over that second AND acts as our loop delay.
            cpu = psutil.cpu_percent(interval=1)
            
            # Since cpu_percent blocks for 1 second, check again if we should exit
            if not self._running:
                break
                
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            battery = psutil.sensors_battery()
            
            # GPU Telemetry
            gpu_data = []
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_data.append({
                            "name": gpu.name,
                            "load": gpu.load * 100,
                            "temperature": gpu.temperature
                        })
                except Exception:
                    pass

            metrics = {
                "timestamp": time.time(),
                "cpu_percent": cpu,
                "cpu_cores": psutil.cpu_count(),
                "ram_percent": ram.percent,
                "ram_used_gb": ram.used / (1024**3),
                "disk_percent": disk.percent,
                "battery_percent": battery.percent if battery else None,
                "battery_plugged": battery.power_plugged if battery else None,
                "gpus": gpu_data
            }
            
            with self._lock:
                self._buffer.append(metrics)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._collect, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            # Note: The thread might take up to 1 second to exit due to cpu_percent(interval=1)
            self._thread.join(timeout=2.0)

    def get_latest(self):
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def get_time_series(self):
        with self._lock:
            return list(self._buffer)

if __name__ == "__main__":
    print("Starting Telemetry Daemon...")
    daemon = SystemTelemetryDaemon(history_seconds=30)
    daemon.start()
    
    print("Gathering data... waiting 5 seconds for initial buffer buildup.")
    time.sleep(5) 
    
    print(f"Data points available: {len(daemon.get_time_series())}")
    print("Most recent data point:")
    print(daemon.get_latest())
    
    print("Stopping daemon...")
    daemon.stop()