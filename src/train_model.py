import argparse
import pickle
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_cluster_data
from src.feature_engineering import compute_tag_frequencies, build_cluster_feature_matrix

def main(cluster_json, model_path):
    clusters = load_cluster_data(cluster_json)
    tag_counter = compute_tag_frequencies(clusters)
    # Optionally, define good/bad tags manually
    good_tags = set()
    bad_tags = set()
    X = build_cluster_feature_matrix(clusters, tag_counter, good_tags, bad_tags)
    y = [cluster['label'] for cluster in clusters]
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Cluster-level model trained and saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_json', type=str, default='../final_groundtruth.json')
    parser.add_argument('--model_path', type=str, default='../models/tag_classifier.pkl')
    args = parser.parse_args()
    main(args.cluster_json, args.model_path) 