import time
import sqlite3
import os
import threading
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gmm_clustering import train_model, load_data, find_database, get_latest_cluster, get_cluster_features
from labeler import label_clusters, load_labels

class EmotionDaemon:
    def __init__(self, db_path="telemetry_aggregated.db", model_dir="models", retrain_interval=5, n_components=5, api_key=None):
        self.db_path = find_database(db_path)
        self.model_dir = model_dir
        self.retrain_interval = retrain_interval
        self.n_components = n_components
        self.running = False
        self.thread = None
        self.last_train_count = 0
        self.current_cluster = []
        self.emotion_labels = load_labels()
        self.api_key = api_key
        
    def _get_row_count(self):
        if not os.path.exists(self.db_path):
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM telemetry")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Database error: {e}")
            return 0
            
    def _loop(self):
        print(f"Emotion Daemon started. Watching {self.db_path}")
        # Initial check
        self.last_train_count = self._get_row_count()
        
        while self.running:
            current_count = self._get_row_count()
            
            # Check if we need to retrain
            if current_count >= self.last_train_count + self.retrain_interval:
                print(f"\n[{time.strftime('%X')}] Row count increased from {self.last_train_count} to {current_count}. Retraining...")
                try:
                    df = load_data(self.db_path)
                    if len(df) >= self.n_components: # Need at least n samples to cluster into n components
                        train_model(df, self.n_components, self.model_dir)
                        self.last_train_count = current_count
                        
                        # After training, label clusters via LLM
                        features = get_cluster_features(self.model_dir)
                        if features:
                            labels = label_clusters(features, api_key=self.api_key)
                            if labels:
                                self.emotion_labels = labels
                    else:
                        print(f"Not enough data to train (Need at least {self.n_components} rows)")
                except Exception as e:
                    print(f"Error during retraining: {e}")
            
            # Update latest cluster
            if current_count > 0 and os.path.exists(os.path.join(self.model_dir, 'gmm_model.pkl')):
                try:
                    self.current_cluster = get_latest_cluster(self.db_path, self.model_dir)
                except Exception as e:
                    print(f"Error predicting latest cluster: {e}")
                    
            time.sleep(5) # Poll every 5 seconds
            
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def get_cluster_characteristics(self, top_n=5):
        """Returns the top distinguishing features for each cluster."""
        return get_cluster_features(self.model_dir, top_n=top_n)
        
    def get_emotion_labels(self):
        """Returns the current emotion label dictionary, e.g. {'0': 'Calm', '1': 'Stressed', ...}."""
        return self.emotion_labels
            
    def get_current_emotion(self):
        """Returns a list of tuples with the cluster IDs and their probability coefficients."""
        return self.current_cluster

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Emotion Daemon.")
    parser.add_argument("--db", type=str, default="telemetry_aggregated.db", help="Path to SQLite DB.")
    parser.add_argument("--interval", type=int, default=5, help="Number of new rows before retraining.")
    args = parser.parse_args()
    
    daemon = EmotionDaemon(db_path=args.db, retrain_interval=args.interval)
    
    try:
        daemon.start()
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            cluster_probs = daemon.get_current_emotion()
            labels = daemon.get_emotion_labels()
            if cluster_probs:
                parts = []
                for c, p in cluster_probs:
                    name = labels.get(str(c), f"Cluster {c}")
                    parts.append(f"{name}: {p*100:.1f}%")
                print(f"[{time.strftime('%X')}] Emotion State: {', '.join(parts)}")
            else:
                print(f"[{time.strftime('%X')}] Emotion State: Unknown")
    except KeyboardInterrupt:
        print("\nStopping Emotion Daemon...")
        daemon.stop()
