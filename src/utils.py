def get_tag_weights(tag_counter, good_tags=None, bad_tags=None):
    weights = {}
    for tag in tag_counter:
        if good_tags and tag in good_tags:
            weights[tag] = 2.0
        elif bad_tags and tag in bad_tags:
            weights[tag] = 0.5
        else:
            weights[tag] = 1.0 + 0.1 * tag_counter[tag]  # frequency-based
    return weights

def aggregate_cluster_predictions(tag_preds):
    """Return 1 if majority of tags are meaningful, else 0."""
    return int(sum(tag_preds) > len(tag_preds) / 2) 