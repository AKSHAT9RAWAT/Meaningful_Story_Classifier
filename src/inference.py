import argparse
import pickle
import numpy as np
from src.feature_engineering import compute_tag_frequencies, build_cluster_feature_matrix

def predict_cluster(model, tags, tag_counter, good_tags=None, bad_tags=None):
    features = build_cluster_feature_matrix([
        {'tags': tags}
    ], tag_counter, good_tags, bad_tags)
    pred = model.predict(features)[0]
    return int(pred)

def main(model_path, input_path, output_path, tags):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    good_tags = set()
    bad_tags = set()
    if tags:
        # Single cluster mode
        tag_counter = compute_tag_frequencies([{'tags': tags, 'label': 1}])
        pred = predict_cluster(model, tags, tag_counter, good_tags, bad_tags)
        print(f"Prediction for cluster: {pred}")
        return
    # Batch mode: input_path is a JSON file with {cluster_id: [tags]}
    import json
    with open(input_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    cluster_list = [{'tags': t} for t in clusters.values()]
    tag_counter = compute_tag_frequencies([{'tags': t, 'label': 1} for t in clusters.values()])
    X = build_cluster_feature_matrix(cluster_list, tag_counter, good_tags, bad_tags)
    preds = model.predict(X)
    results = {cid: int(pred) for cid, pred in zip(clusters.keys(), preds)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Batch inference complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/tag_classifier.pkl')
    parser.add_argument('--input_path', type=str, default=None, help='JSON file with {cluster_id: [tags]}')
    parser.add_argument('--output_path', type=str, default='../cluster_inference_results.json')
    parser.add_argument('--tags', nargs='+', default=None, help='List of tags for a single cluster')
    args = parser.parse_args()
    main(args.model_path, args.input_path, args.output_path, args.tags) 