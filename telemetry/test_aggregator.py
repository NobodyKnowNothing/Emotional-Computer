import unittest
import numpy as np
import time
import os
import pandas as pd
from aggregator import TelemetryAggregator

class TestTelemetryAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = TelemetryAggregator(interval_seconds=1, db_file="test_agg_tmp.db")

    def tearDown(self):
        if os.path.exists("test_agg_tmp.db"):
            try:
                os.remove("test_agg_tmp.db")
            except PermissionError:
                pass

    def test_flatten_metrics(self):
        metrics = {
            "timestamp": 12345.0,
            "cpu_cores": 8,
            "battery_plugged": True,
            "cpu_percent": 50.5,
            "ram_percent": 60.0,
            "gpus": [
                {"load": 30.0, "temperature": 40.0},
                {"load": 10.0, "temperature": 35.0}
            ],
            "missing_metric": None
        }
        
        flat = self.aggregator._flatten_metrics(metrics)
        
        # Check ignored fields
        self.assertNotIn("timestamp", flat)
        self.assertNotIn("cpu_cores", flat)
        self.assertNotIn("battery_plugged", flat)
        
        # Check standard fields
        self.assertEqual(flat["cpu_percent"], 50.5)
        self.assertEqual(flat["ram_percent"], 60.0)
        
        # Check GPU flattening
        self.assertEqual(flat["gpu_0_load"], 30.0)
        self.assertEqual(flat["gpu_0_temperature"], 40.0)
        self.assertEqual(flat["gpu_1_load"], 10.0)
        self.assertEqual(flat["gpu_1_temperature"], 35.0)
        
        # Check None handled as NaN
        self.assertTrue(np.isnan(flat["missing_metric"]))

    def test_calculate_stats(self):
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        stats = self.aggregator.calculate_stats(data, timestamps)
        
        self.assertAlmostEqual(stats["mean"], 30.0)
        self.assertAlmostEqual(stats["median"], 30.0)
        self.assertTrue(stats["variance"] > 0)
        # Slope should be 10.0 per second
        self.assertAlmostEqual(stats["slope"], 10.0)
        
    def test_calculate_stats_with_nan(self):
        timestamps = [0.0, 1.0, 2.0]
        data = [10.0, np.nan, 30.0]
        
        stats = self.aggregator.calculate_stats(data, timestamps)
        
        self.assertAlmostEqual(stats["mean"], 20.0)
        self.assertAlmostEqual(stats["median"], 20.0)
        self.assertAlmostEqual(stats["slope"], 10.0) # (30-10) / (2-0)
        
    def test_process_buffer(self):
        buffer = [
            {"timestamp": 100.0, "cpu_percent": 10.0},
            {"timestamp": 101.0, "cpu_percent": 20.0},
            {"timestamp": 102.0, "cpu_percent": 30.0}
        ]
        
        log_stats = {"total_logs": 5, "warnings": 1, "errors": 0}
        
        row = self.aggregator.process_buffer(buffer, log_stats)
        
        self.assertEqual(row["timestamp_end"], 102.0)
        self.assertAlmostEqual(row["cpu_percent_mean"], 20.0)
        self.assertAlmostEqual(row["cpu_percent_slope"], 10.0)
        self.assertEqual(row["logs_total"], 5)
        self.assertEqual(row["logs_warnings"], 1)

    def test_save_to_db(self):
        import sqlite3
        row1 = {"timestamp_end": 100.0, "cpu_percent_mean": 20.0}
        row2 = {"timestamp_end": 101.0, "cpu_percent_mean": 30.0}
        
        self.aggregator.save_to_db(row1)
        self.aggregator.save_to_db(row2)
        
        with sqlite3.connect("test_agg_tmp.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp_end, cpu_percent_mean FROM telemetry ORDER BY timestamp_end")
            rows = cursor.fetchall()
            
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], 20.0)
        self.assertEqual(rows[1][1], 30.0)

if __name__ == "__main__":
    unittest.main()
