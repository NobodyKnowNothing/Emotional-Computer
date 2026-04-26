import os
import sys
from fastmcp import FastMCP

# Add subdirectories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "telemetry"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emotions"))

from aggregator import TelemetryAggregator
from daemon import EmotionDaemon

# Initialize FastMCP server
mcp = FastMCP("Emotional Computer")

# Global instances
db_path = "telemetry_aggregated.db"
emotion_daemon = EmotionDaemon(db_path=db_path)
aggregator = TelemetryAggregator(db_file=db_path)

@mcp.resource("emotion://current")
def get_current_emotion() -> str:
    """Returns the current emotional state of the computer."""
    probs = emotion_daemon.get_current_emotion()
    labels = emotion_daemon.get_emotion_labels()
    
    if not probs:
        return "The computer is currently emotionless (no data yet)."
    
    # Format the top emotion
    top_cid, top_prob = probs[0]
    label = labels.get(str(top_cid), f"Cluster {top_cid}")
    
    # Build a descriptive string
    description = f"Current State: {label} ({top_prob*100:.1f}% confidence)\n"
    description += "Full Distribution:\n"
    for cid, p in probs:
        l = labels.get(str(cid), f"Cluster {cid}")
        description += f"- {l}: {p*100:.1f}%\n"
        
    return description

@mcp.tool()
def get_system_metrics() -> dict:
    """Returns the raw latest telemetry metrics."""
    return aggregator.daemon.get_latest()

# On startup, start our background daemons
@mcp.on_startup()
def start_daemons():
    print("Starting background telemetry and emotion daemons...")
    emotion_daemon.start()
    aggregator.daemon.start()
    aggregator.log_counter.start()
    # We don't call aggregator.run() because it blocks; 
    # instead we let the MCP server handle the main loop.
    # We'll manually trigger aggregation in a background thread if needed,
    # or just rely on the daemon's rolling buffers for tools.

if __name__ == "__main__":
    mcp.run()