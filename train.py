# import json
# import torch
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# # Load your data
# with open("final_groundtruth.json", "r") as f:
#     data = json.load(f)

# # Step 1: Build tag vocabulary
# all_tags = set(tag for album in data.values() for tag in album["tags"])
# tag2idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
# vocab_size = len(tag2idx)

# # Step 2: Create Dataset Class
# class TagDataset(Dataset):
#     def __init__(self, data_dict):
#         self.samples = []
#         for entry in data_dict.values():
#             tag_indices = [tag2idx[tag] for tag in entry["tags"] if tag in tag2idx]
#             self.samples.append((tag_indices, entry["label"]))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# # Step 3: Collate function for batching
# def collate_fn(batch):
#     tag_lists, labels = zip(*batch)
#     padded_tags = pad_sequence([torch.tensor(t) for t in tag_lists], batch_first=True)
#     return padded_tags, torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# # Step 4: Neural model
# class TagMLP(nn.Module):
#     def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, tag_indices):
#         # Get average tag embedding
#         tag_embeds = self.embedding(tag_indices)  # [batch, seq, dim]
#         mean_embed = tag_embeds.mean(dim=1)       # [batch, dim]
#         return self.mlp(mean_embed)

# # Step 5: Prepare data
# dataset = TagDataset(data)
# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

# # Step 6: Train
# model = TagMLP(vocab_size)
# loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(10):
#     model.train()
#     total_loss = 0
#     for tags, labels in train_loader:
#         preds = model(tags)
#         loss = loss_fn(preds, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# # Step 7: Evaluate
# model.eval()
# y_true, y_pred = [], []
# with torch.no_grad():
#     for tags, labels in test_loader:
#         preds = model(tags)
#         pred_classes = (preds > 0.5).int().squeeze().tolist()
#         y_pred.extend(pred_classes)
#         y_true.extend(labels.squeeze().tolist())

# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred))


import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---------------- Configuration ----------------
EMBED_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MODEL_PATH = "tag_model.pt"
VOCAB_PATH = "tag2idx.json"
DATA_PATH = "final_groundtruth.json"  # <-- Put your dataset here

# ---------------- Dataset and Vocab ----------------
class TagDataset(Dataset):
    def __init__(self, data_dict, tag2idx):
        self.samples = []
        for entry in data_dict.values():
            tag_indices = [tag2idx[tag] for tag in entry["tags"] if tag in tag2idx]
            self.samples.append((tag_indices, entry["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    tag_lists, labels = zip(*batch)
    padded = pad_sequence([torch.tensor(t) for t in tag_lists], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded, labels

# ---------------- Model ----------------
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
        embeds = self.embedding(tag_indices)      # [batch, seq, dim]
        mean_embed = embeds.mean(dim=1)           # [batch, dim]
        return self.mlp(mean_embed)

# ---------------- Main Training ----------------
if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # Build vocabulary
    all_tags = set(tag for d in data.values() for tag in d["tags"])
    tag2idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
    with open(VOCAB_PATH, "w") as f:
        json.dump(tag2idx, f)

    # Prepare datasets
    dataset = TagDataset(data, tag2idx)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TagMLP(len(tag2idx), EMBED_DIM, HIDDEN_DIM)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for tags, labels in train_loader:
            preds = model(tags)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for tags, labels in test_loader:
            preds = model(tags)
            pred_classes = (preds > 0.5).int().squeeze().tolist()
            y_pred.extend(pred_classes)
            y_true.extend(labels.squeeze().tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
