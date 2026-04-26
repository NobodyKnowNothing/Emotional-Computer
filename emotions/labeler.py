import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import get_config_context

from google import genai

LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_labels.json")

def _build_prompt(cluster_features):
    """Builds the prompt describing each cluster's top features for the LLM."""
    lines = [
        "You are analyzing system telemetry clusters from a computer.",
        "Each cluster represents a distinct behavioral state of the machine.",
        "Based on the top distinguishing features (shown as z-score deviations from the global mean),",
        "assign each cluster a short, evocative emotion label (1-3 words) that personifies the computer's state.",
        "",
        "Here are the clusters and their top features:",
        "",
    ]
    
    for cluster_id, features in cluster_features.items():
        feature_desc = ", ".join([f"{name}: {z:.2f}" for name, z in features])
        lines.append(f"Cluster {cluster_id}: [{feature_desc}]")
    
    lines.extend([
        "",
        "Respond with ONLY a valid JSON object mapping cluster integer IDs (as strings) to their emotion label strings.",
        "Do not include any other text, markdown formatting, or code fences.",
        'Example format: {"0": "[EMOTION]", "1": "[EMOTION]", "2": "[EMOTION]", "3": "[EMOTION]", "4": "[EMOTION]"}',
    ])
    
    return "\n".join(lines)

def _validate_labels(raw_text, expected_cluster_ids):
    """Parses and validates the LLM response as a proper {str(int): str} JSON dict."""
    # Strip markdown code fences if the LLM wraps its response
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove opening fence (e.g. ```json)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"Labeler: LLM response was not valid JSON: {raw_text[:200]}")
        return None
        
    if not isinstance(parsed, dict):
        print(f"Labeler: Expected a JSON object, got {type(parsed).__name__}")
        return None
        
    # Normalize keys to strings and validate values are strings
    validated = {}
    for key, value in parsed.items():
        try:
            int_key = int(key)
        except (ValueError, TypeError):
            print(f"Labeler: Key '{key}' is not a valid integer.")
            return None
            
        if not isinstance(value, str) or not value.strip():
            print(f"Labeler: Value for cluster {int_key} must be a non-empty string, got: {value}")
            return None
            
        validated[str(int_key)] = value.strip()
    
    # Verify all expected cluster IDs are present
    expected_keys = {str(cid) for cid in expected_cluster_ids}
    missing = expected_keys - set(validated.keys())
    if missing:
        print(f"Labeler: Missing labels for clusters: {missing}")
        return None
        
    return validated

def label_clusters(cluster_features, api_key=None):
    """
    Sends cluster features to the Gemini LLM and asks it to assign emotion labels.
    Returns a dict like {"0": "Calm", "1": "Stressed", ...} or None on failure.
    Saves the result to emotion_labels.json.
    """
    if not cluster_features:
        print("Labeler: No cluster features provided.")
        return None
    
    # Resolve API key
    if api_key is None:
        try:
            config_context = get_config_context()
            api_key = config_context.get("env", {}).get("GEMINI_API_KEY")
        except Exception:
            pass
            
    if not api_key:
        print("Labeler: No GEMINI_API_KEY found. Skipping label generation.")
        return None
    
    prompt = _build_prompt(cluster_features)
    expected_ids = list(cluster_features.keys())
    
    gemini_client = genai.Client(api_key=api_key)
    
    # Try up to 2 times in case the LLM returns bad formatting on the first attempt
    for attempt in range(2):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                ),
            )
        except Exception as e:
            print(f"Labeler: LLM API call failed: {e}")
            return None
            
        raw_text = response.text
        labels = _validate_labels(raw_text, expected_ids)
        
        if labels is not None:
            # Save to local JSON
            try:
                with open(LABELS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(labels, f, indent=2)
                print(f"Labeler: Saved emotion labels to {LABELS_FILE}")
            except Exception as e:
                print(f"Labeler: Failed to write labels file: {e}")
                
            return labels
        else:
            print(f"Labeler: Attempt {attempt + 1} failed validation, retrying...")
            
    print("Labeler: Could not get valid labels after 2 attempts.")
    return None

def load_labels():
    """Loads the most recently saved emotion labels from disk. Returns dict or empty dict."""
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}
