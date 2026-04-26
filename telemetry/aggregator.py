import os
import time
import sys
import argparse
import sqlite3
from telemetry import SystemTelemetryDaemon
from logs import SystemLogCounter

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required.")
    print("Please install it using: pip install numpy")
    sys.exit(1)

class TelemetryAggregator:
    def __init__(self, interval_seconds=30, db_file="telemetry_aggregated.db"):
        self.interval_seconds = interval_seconds
        self.db_file = db_file
        # Keep a larger buffer (2x interval) to ensure no data points are dropped
        # between polling cycles.
        self.daemon = SystemTelemetryDaemon(history_seconds=interval_seconds * 2)
        self.log_counter = SystemLogCounter(window_seconds=interval_seconds)
        self.last_timestamp = time.time()
        
    def _flatten_metrics(self, metrics):
        """Flattens nested dictionaries (like GPUs) into a single-level dictionary."""
        flat = {}
        for key, value in metrics.items():
            if key in ("timestamp", "cpu_cores", "battery_plugged"):
                continue  # Skip timestamp (handled separately) and static/boolean fields
                
            if key == "gpus" and isinstance(value, list):
                for i, gpu in enumerate(value):
                    flat[f"gpu_{i}_load"] = gpu.get("load", np.nan)
                    flat[f"gpu_{i}_temperature"] = gpu.get("temperature", np.nan)
            elif key == "fans" and isinstance(value, list):
                for i, fan in enumerate(value):
                    flat[f"fan_{i}_speed"] = fan.get("current", np.nan)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[key] = value
            elif value is None:
                flat[key] = np.nan
        return flat

    def calculate_stats(self, series_data, timestamps, full_stats=True):
        """Calculates mean, median, variance, and slope for a given time series."""
        if not series_data:
            return {"mean": np.nan, "median": np.nan, "variance": np.nan, "slope": np.nan}
            
        data = np.array(series_data, dtype=float)
        # Filter out NaN values (e.g. if battery was unplugged/missing for a moment)
        valid_idx = ~np.isnan(data)
        if not np.any(valid_idx):
            return {"mean": np.nan, "median": np.nan, "variance": np.nan, "slope": np.nan}
            
        valid_data = data[valid_idx]
        valid_timestamps = np.array(timestamps)[valid_idx]
        
        median = float(np.median(valid_data))
        
        if full_stats:
            mean = float(np.mean(valid_data))
            var = float(np.var(valid_data))
            is_constant = (var == 0.0)
        else:
            mean = np.nan
            var = np.nan
            is_constant = (float(np.max(valid_data)) == float(np.min(valid_data)))
        
        # Calculate slope (average unit change per second) using simple linear regression
        if is_constant:
            slope = 0.0
        elif len(valid_data) > 1 and valid_timestamps[-1] != valid_timestamps[0]:
            # Shift timestamps to start at 0 to avoid float precision issues
            t = valid_timestamps - valid_timestamps[0]
            # np.polyfit(x, y, 1) returns [slope, intercept]
            slope = float(np.polyfit(t, valid_data, 1)[0])
        else:
            slope = 0.0
            
        return {"mean": mean, "median": median, "variance": var, "slope": slope}

    def process_buffer(self, buffer, log_stats=None):
        """Processes a list of telemetry dictionaries into a single row of stats."""
        if not buffer:
            if not log_stats:
                return None
            row = {"timestamp_end": time.time()}
        else:
            timestamps = [d["timestamp"] for d in buffer]
            flattened_buffer = [self._flatten_metrics(d) for d in buffer]
            
            # Collect all unique keys that appeared in this time window
            all_keys = set()
            for f in flattened_buffer:
                all_keys.update(f.keys())
                
            # Start row with the end timestamp (representing this aggregated block)
            row = {"timestamp_end": timestamps[-1]}
            
            for key in all_keys:
                series_data = [f.get(key, np.nan) for f in flattened_buffer]
                
                if key == "battery_percent":
                    stats = self.calculate_stats(series_data, timestamps, full_stats=False)
                    row[f"{key}_median"] = stats["median"]
                    row[f"{key}_slope"] = stats["slope"]
                else:
                    stats = self.calculate_stats(series_data, timestamps)
                    row[f"{key}_mean"] = stats["mean"]
                    row[f"{key}_median"] = stats["median"]
                    row[f"{key}_variance"] = stats["variance"]
                    row[f"{key}_slope"] = stats["slope"]
                
        if log_stats:
            row["logs_total"] = log_stats.get("total_logs", 0)
            row["logs_warnings"] = log_stats.get("warnings", 0)
            row["logs_errors"] = log_stats.get("errors", 0)
            
        return row

    def save_to_db(self, row):
        """Appends the new row to the SQLite database, handling dynamic columns safely."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    timestamp_end REAL PRIMARY KEY
                )
            ''')
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(telemetry)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            # Add missing columns
            for key, value in row.items():
                if key not in existing_columns:
                    col_type = "REAL" if isinstance(value, float) else "INTEGER"
                    try:
                        cursor.execute(f"ALTER TABLE telemetry ADD COLUMN {key} {col_type}")
                        existing_columns.add(key)
                    except sqlite3.OperationalError as e:
                        print(f"Warning: Could not add column {key} ({e})")
            
            # Filter row to only include keys that exist in the table and convert NaN to None
            filtered_row = {
                k: (None if isinstance(v, float) and np.isnan(v) else v) 
                for k, v in row.items() if k in existing_columns
            }
            
            cols = ', '.join(filtered_row.keys())
            placeholders = ', '.join(['?'] * len(filtered_row))
            
            try:
                cursor.execute(f"INSERT OR REPLACE INTO telemetry ({cols}) VALUES ({placeholders})", tuple(filtered_row.values()))
            except sqlite3.Error as e:
                print(f"Error inserting row into database: {e}")

    def run(self):
        """Runs the main continuous aggregation loop."""
        print("Starting Telemetry Daemon...")
        self.daemon.start()
        print("Starting System Log Counter...")
        self.log_counter.start()
        print(f"Aggregating telemetry every {self.interval_seconds} seconds.")
        print(f"Output will be saved to: {os.path.abspath(self.db_file)}")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                # Wait for the specified interval
                time.sleep(self.interval_seconds)
                
                # Fetch rolling buffer
                buffer = self.daemon.get_time_series()
                
                # Fetch log stats for the interval
                log_stats = self.log_counter.get_last_30s_stats()
                
                # Filter points we haven't processed yet
                new_data = [d for d in buffer if d["timestamp"] > self.last_timestamp]
                
                if new_data or log_stats:
                    if new_data:
                        self.last_timestamp = new_data[-1]["timestamp"]
                    row = self.process_buffer(new_data, log_stats)
                    if row:
                        self.save_to_db(row)
                        print(f"[{time.strftime('%X')}] Aggregated {len(new_data)} data points & logs -> {self.db_file}")
                else:
                    print(f"[{time.strftime('%X')}] No new data points or logs to aggregate.")
                    
        except KeyboardInterrupt:
            print("\nStopping aggregator...")
        finally:
            self.daemon.stop()
            self.log_counter.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates system telemetry into an SQLite database.")
    parser.add_argument("--interval", type=int, default=30, help="Aggregation interval in seconds (default: 30).")
    parser.add_argument("--db", type=str, default="telemetry_aggregated.db", help="Output SQLite database file path.")
    
    args = parser.parse_args()
    
    aggregator = TelemetryAggregator(interval_seconds=args.interval, db_file=args.db)
    aggregator.run()
