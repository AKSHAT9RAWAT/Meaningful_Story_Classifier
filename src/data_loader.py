import json
from pathlib import Path

def load_cluster_data(cluster_json_path):
    """Load cluster tags and labels from final_groundtruth.json."""
    with open(cluster_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    clusters = []
    for cluster_id, cluster_info in data.items():
        clusters.append({
            'cluster_id': cluster_id,
            'tags': cluster_info['tags'],
            'label': cluster_info['label']
        })
    return clusters

def load_individual_tags(individual_json_path):
    """Load individual image tags from full_individual_tags.json."""
    with open(individual_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expecting: {image_id: [tag1, tag2, ...], ...}
    return data 