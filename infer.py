import json
import torch
import torch.nn as nn
from model import TagMLP  # Make sure model.py is in the same folder

# --------------- Config ---------------
EMBED_DIM = 32
HIDDEN_DIM = 64
MODEL_PATH = "tag1_model.pt"         # <- Adjust if needed
VOCAB_PATH = "tag2idx1.json"         # <- Adjust if needed
THRESHOLD = 0.95

# --------------- Inference Function ---------------
def infer_batch(tag_clusters):
    with open(VOCAB_PATH, "r") as f:
        tag2idx = json.load(f)

    vocab_size = len(tag2idx)
    model = TagMLP(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    all_tensors = []
    original_indices = []

    for i, tag_list in enumerate(tag_clusters):
        indices = [tag2idx[tag] for tag in tag_list if tag in tag2idx]
        if indices:
            all_tensors.append(torch.tensor(indices))
            original_indices.append(i)
        else:
            print(f"Album {i}: No known tags in input.")

    if not all_tensors:
        print("No valid albums to infer.")
        return

    # Pad and infer
    padded = torch.nn.utils.rnn.pad_sequence(all_tensors, batch_first=True)
    with torch.no_grad():
        preds = model(padded).view(-1)  # 👈 fixes the 0-d tensor issue

    for i, pred in zip(original_indices, preds):
        score = pred.item()
        label = 1 if score > THRESHOLD else 0
        print(f"Album {i} — Prediction: {label} (Score: {score:.4f})")

# --------------- Example Usage ---------------
if __name__ == "__main__":
    tag_clusters = [
   ["family", "picnic", "park", "blanket", "children", "sunny"],
    ["sports", "team", "stadium", "crowd", "cheering", "match"],
    ["dog", "fence", "yard", "tree", "grass"],
    ["flower", "pot", "window", "curtain", "vase"],
    ["shelf", "boxes", "label", "inventory", "light"]
    ]

    infer_batch(tag_clusters)
