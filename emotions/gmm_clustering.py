import sqlite3
import argparse
import sys
import os
import pickle
import numpy as np

try:
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: pandas and scikit-learn are required.")
    print("Please install them using: pip install pandas scikit-learn")
    sys.exit(1)

def find_database(provided_path):
    """Attempt to find the database file in a few common locations."""
    if os.path.exists(provided_path):
        return provided_path
    
    parent_path = os.path.join("..", provided_path)
    if os.path.exists(parent_path):
        return parent_path
        
    telemetry_path = os.path.join("..", "telemetry", provided_path)
    if os.path.exists(telemetry_path):
        return telemetry_path

    return provided_path

def load_data(db_path, limit=None):
    """Loads telemetry data from SQLite database into a pandas DataFrame."""
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM telemetry ORDER BY timestamp_end DESC"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        # Reverse to chronological order if we pulled a limit
        if limit:
             df = df.iloc[::-1].reset_index(drop=True)
    finally:
        conn.close()
    return df

def train_model(df, n_components, model_dir):
    """Trains the GMM and Scaler, then saves them for live inference."""
    print("--- Training Mode ---")
    feature_cols = [col for col in df.columns if col != 'timestamp_end']
    if not feature_cols:
        print("No feature columns found in the database.")
        sys.exit(1)
        
    X = df[feature_cols].copy()
    
    # Handle missing values and store the means to use during live inference
    mean_fill = X.mean()
    X = X.fillna(mean_fill).fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full')
    labels = gmm.fit_predict(X_scaled)
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'gmm_model.pkl'), 'wb') as f:
        pickle.dump(gmm, f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
        pickle.dump({'columns': feature_cols, 'means': mean_fill}, f)
        
    print(f"Models successfully trained and saved to '{model_dir}'")
    
    return labels, feature_cols

def predict_latest(df, model_dir):
    """Loads models and predicts the cluster for the provided data (e.g. latest log)."""
    print("--- Inference (Live) Mode ---")
    try:
        with open(os.path.join(model_dir, 'gmm_model.pkl'), 'rb') as f:
            gmm = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            meta = pickle.load(f)
            feature_cols = meta['columns']
            mean_fill = meta['means']
    except FileNotFoundError:
        print(f"Error: Models not found in {model_dir}. Please run with '--mode train' first.")
        sys.exit(1)
        
    # Ensure we use exactly the features the model was trained on
    # Add missing columns with NaN
    missing_cols = set(feature_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = np.nan
        
    X = df[feature_cols].copy()
    
    # Fill missing values using the saved training means
    X = X.fillna(mean_fill).fillna(0)
    
    # Apply the saved scaler
    X_scaled = scaler.transform(X)
    
    # Predict clusters and probabilities
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    for i, row in df.iterrows():
        timestamp = row.get('timestamp_end', 'Unknown')
        cluster = labels[i]
        confidence = probs[i][cluster] * 100
        
        print(f"\nEvaluating log at Timestamp: {timestamp}")
        print(f"Assigned Cluster: {cluster} (Confidence: {confidence:.1f}%)")
        
        if confidence < 50.0:
            print("WARNING: Low confidence! This point might be an anomaly or a new state.")
            
        print("Probability Distribution:")
        for c_idx, p in enumerate(probs[i]):
            print(f"  Cluster {c_idx}: {p*100:.1f}%")
            
    return df_clustered

def get_latest_cluster(db_path, model_dir):
    """Quietly returns the cluster participation coefficients for the most recent telemetry entry.
    Returns a list of tuples: [(cluster_id, probability), ...]
    """
    df = load_data(db_path, limit=1)
    if df.empty:
        return []
        
    try:
        with open(os.path.join(model_dir, 'gmm_model.pkl'), 'rb') as f:
            gmm = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            meta = pickle.load(f)
            feature_cols = meta['columns']
            mean_fill = meta['means']
    except FileNotFoundError:
        return []
        
    missing_cols = set(feature_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = np.nan
        
    X = df[feature_cols].copy()
    X = X.fillna(mean_fill).fillna(0)
    X_scaled = scaler.transform(X)
    
    probs = gmm.predict_proba(X_scaled)[0]
    
    # Return as [(cluster_id, probability_coefficient), ...] sorted by highest prob
    participation = [(i, float(p)) for i, p in enumerate(probs)]
    participation.sort(key=lambda x: x[1], reverse=True)
    
    return participation

def get_cluster_features(model_dir, top_n=5):
    """
    Returns the top distinguishing features for each cluster based on the GMM's learned means.
    Returns a dictionary mapping cluster_id to a list of (feature_name, deviation_z_score) tuples.
    """
    try:
        with open(os.path.join(model_dir, 'gmm_model.pkl'), 'rb') as f:
            gmm = pickle.load(f)
        with open(os.path.join(model_dir, 'features.pkl'), 'rb') as f:
            meta = pickle.load(f)
            feature_cols = meta['columns']
    except FileNotFoundError:
        return {}

    cluster_features = {}
    
    # gmm.means_ has shape (n_components, n_features). 
    # Since data was StandardScaled, these are z-scores (deviations from global mean).
    for cluster_id, means in enumerate(gmm.means_):
        # Pair feature names with their mean z-score for this cluster
        feature_deviations = [(feature_cols[i], float(m)) for i, m in enumerate(means)]
        
        # Sort by absolute deviation (how uniquely high or low this feature is for this cluster)
        feature_deviations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        cluster_features[cluster_id] = feature_deviations[:top_n]
        
    return cluster_features

def analyze_clusters(df, labels, feature_cols):
    """Analyzes and prints statistics for the generated clusters."""
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    print(f"\n--- Clustering Summary (Training) ---")
    print(f"Total data points: {len(df_clustered)}")
    print("\nCluster Distribution:")
    print(df_clustered['cluster'].value_counts().sort_index())
    
    print("\nCluster Means (Top 5 features by variance):")
    # Find features with highest variance to show meaningful differences
    variances = df_clustered[feature_cols].var()
    top_features = variances.nlargest(5).index.tolist()
    
    if top_features:
        means = df_clustered.groupby('cluster')[top_features].mean()
        print(means)
    
    return df_clustered

def main():
    parser = argparse.ArgumentParser(description="Run GMM clustering and live inference on telemetry data.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="'train' to build/save models on full DB, 'predict' to classify recent logs.")
    parser.add_argument("--db", type=str, default="telemetry_aggregated.db", help="Path or name of the SQLite database.")
    parser.add_argument("--components", type=int, default=5, help="Number of GMM components (clusters) - only used in train mode.")
    parser.add_argument("--output", type=str, default="clustered_telemetry.csv", help="Optional CSV output file for clustered data.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save/load the trained models.")
    parser.add_argument("--limit", type=int, default=1, help="Number of recent logs to evaluate in predict mode (default: 1).")
    
    args = parser.parse_args()
    
    db_path = find_database(args.db)
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'.")
        print("Please ensure the aggregator has run and generated data.")
        sys.exit(1)
        
    print(f"Loading database: {db_path}")
    
    try:
        # If predicting, we only need the latest N rows. If training, we load all.
        limit = args.limit if args.mode == "predict" else None
        df = load_data(db_path, limit=limit)
    except Exception as e:
        print(f"Error reading database: {e}")
        sys.exit(1)
        
    if df.empty:
        print("Error: The 'telemetry' table is empty.")
        sys.exit(1)
        
    if args.mode == "train":
        print(f"Loaded {len(df)} total rows for training.")
        labels, feature_cols = train_model(df, args.components, args.model_dir)
        df_result = analyze_clusters(df, labels, feature_cols)
        
        if args.output:
            try:
                df_result.to_csv(args.output, index=False)
                print(f"\nSaved clustered training data to: {os.path.abspath(args.output)}")
            except Exception as e:
                print(f"\nError saving to CSV: {e}")
                
    elif args.mode == "predict":
        print(f"Loaded {len(df)} recent row(s) for live inference.")
        df_result = predict_latest(df, args.model_dir)

if __name__ == "__main__":
    main()
