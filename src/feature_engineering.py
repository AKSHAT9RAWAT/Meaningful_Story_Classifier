from collections import Counter
import numpy as np

def compute_tag_frequencies(clusters):
    tag_counter = Counter()
    for cluster in clusters:
        tag_counter.update(cluster['tags'])
    return tag_counter

def tag_length(tag):
    return len(tag)

def is_in_list(tag, tag_list):
    return int(tag in tag_list)

def build_feature_matrix(tags, tag_counter, good_tags=None, bad_tags=None):
    X = []
    for tag in tags:
        features = [
            tag_counter[tag],
            tag_length(tag),
            is_in_list(tag, good_tags) if good_tags else 0,
            is_in_list(tag, bad_tags) if bad_tags else 0
        ]
        X.append(features)
    return np.array(X)

def cluster_num_unique_tags(tags):
    return len(set(tags))

def cluster_avg_tag_frequency(tags, tag_counter):
    return sum(tag_counter[tag] for tag in tags) / len(tags) if tags else 0

def cluster_fraction_in_list(tags, tag_list):
    if not tags:
        return 0
    return sum(1 for tag in tags if tag in tag_list) / len(tags)

def cluster_avg_tag_length(tags):
    return sum(len(tag) for tag in tags) / len(tags) if tags else 0

def cluster_tag_diversity(tags):
    return len(set(tags)) / len(tags) if tags else 0

def build_cluster_feature_matrix(clusters, tag_counter, good_tags=None, bad_tags=None):
    X = []
    for cluster in clusters:
        tags = cluster['tags']
        features = [
            cluster_num_unique_tags(tags),
            cluster_avg_tag_frequency(tags, tag_counter),
            cluster_fraction_in_list(tags, good_tags) if good_tags else 0,
            cluster_fraction_in_list(tags, bad_tags) if bad_tags else 0,
            cluster_avg_tag_length(tags),
            cluster_tag_diversity(tags)
        ]
        X.append(features)
    return np.array(X) 