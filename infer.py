import json
import torch
import torch.nn as nn

# --------------- Config ---------------
EMBED_DIM = 32
HIDDEN_DIM = 64
MODEL_PATH = "tag_model.pt"
VOCAB_PATH = "tag2idx.json"

# --------------- Model ---------------
class TagMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tag_indices):
        embeds = self.embedding(tag_indices)
        mean_embed = embeds.mean(dim=1)
        return self.mlp(mean_embed)

# --------------- Inference ---------------
def infer(tag_list):
    with open(VOCAB_PATH, "r") as f:
        tag2idx = json.load(f)

    vocab_size = len(tag2idx)
    model = TagMLP(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    tag_indices = [tag2idx[tag] for tag in tag_list if tag in tag2idx]
    if not tag_indices:
        print("No known tags in input.")
        return

    input_tensor = torch.tensor(tag_indices).unsqueeze(0)  # [1, seq]
    with torch.no_grad():
        pred = model(input_tensor)
        score = pred.item()
        label = 1 if score > 0.5 else 0
        print(f"Prediction: {label} (Score: {score:.4f})")

# --------------- Example Usage ---------------
if __name__ == "__main__":
    test_tags = [
    "'",
    "each",
    "film",
    "food",
    "group",
    "large",
    "meat",
    "men",
    "metal",
    "next",
    "noodles",
    "object",
    "other",
    "people",
    "plate",
    "sitting",
    "standing",
    "table",
    "team",
    "tent",
    "two",
    "umbrella",
    "vegetables",
    "white"
  ]
    infer(test_tags)
