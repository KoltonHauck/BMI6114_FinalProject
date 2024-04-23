import random
import pickle
import numpy as np

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import F1Score

import lightning as L

class PatientDataset(Dataset):
    def __init__(self, data, text2embedding, undersample=True):
        self.text2embedding = text2embedding
        if undersample:
            self.data = self.undersample_data(data)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient = self.data[idx]
        embeddings = []

        for encounter in patient["encounters"]:
            # Fetch embeddings for each text piece
            for key in ["Description", "ReasonDescription"]:
                text = encounter["encounter"].get(key, "")
                if text and text in self.text2embedding:
                    embeddings.append(self.text2embedding[text])

            for item_type in ["conditions", "careplans", "procedures"]:
                for item in encounter[item_type]:
                    for key in ["Description", "ReasonDescription"]:
                        text = item.get(key, "")
                        if text and text in self.text2embedding:
                            embeddings.append(self.text2embedding[text])

        # Stack embeddings if not empty, else return a zero tensor
        embeddings_tensor = torch.stack(embeddings) if embeddings else torch.zeros(1, len(next(iter(self.text2embedding.values()))))
    
        return {
            "embeddings": embeddings_tensor,  # a list of tensors
            "features": torch.tensor([patient["lat"], patient["lon"]]),
            "label": int(patient["label"]),
            "patient_id": patient["patient_id"]
        }
    
    def undersample_data(self, data):
        # Count instances per class
        label_counts = Counter(patient["label"] for patient in data)
        
        # Find the minimum number of instances in any class
        min_count = min(label_counts.values())
        
        # Create a new list of data with balanced classes
        undersampled_data = []
        counts = {label: 0 for label in label_counts}  # track counts per class
        for patient in data:
            label = patient["label"]
            if counts[label] < min_count:
                undersampled_data.append(patient)
                counts[label] += 1
        return undersampled_data

def custom_pad(data):
    """Pad data which is a list of dictionaries or lists of tensors."""
    # Check if the data is a list of dictionaries
    if all(isinstance(item, dict) for item in data):
        padded_data = {}
        # Iterate through keys in the dictionary
        for key in data[0]:
            # Recurse on the list of items for this key across all dictionaries
            padded_data[key] = custom_pad([d[key] for d in data])
    elif all(isinstance(item, torch.Tensor) for item in data):
        # If the list contains tensors, pad them
        padded_data = pad_sequence(data, batch_first=True)
    else:
        raise TypeError("Unsupported data type for padding")
    return padded_data

def collate_fn(batch):
    embeddings = [item["embeddings"] for item in batch]
    features = torch.stack([item["features"] for item in batch])
    labels = torch.tensor([int(item["label"]) for item in batch])
    ids = [item["patient_id"] for item in batch]

    # Pad the embeddings sequences
    embeddings_padded = pad_sequence(embeddings, batch_first=True)

    return {
        "embeddings": embeddings_padded,
        "features": features,
        "label": labels,
        "patient_id": ids
    }

def split_list(input_list, split_ratios):
    total_length = len(input_list)
    assert sum(split_ratios) == 1.0, "Split ratios must sum up to 1.0"

    split_lengths = [int(total_length * ratio) for ratio in split_ratios]
    split_lengths[-1] += total_length - sum(split_lengths)  # Adjust last split length to handle rounding errors

    splits = []
    start_idx = 0
    for length in split_lengths:
        splits.append(input_list[start_idx:start_idx + length])
        start_idx += length

    return splits

def dataloader_to_numpy(dataloader, pooling="mean"):
    """Convert a PyTorch DataLoader to NumPy arrays with optional pooling."""
    X_list, y_list = [], []
    
    # Loop over the DataLoader
    for batch in dataloader:
        X, y = batch["embeddings"], batch["label"]
        
        # Check the desired pooling method
        if pooling == "mean":
            X = X.mean(dim=1)  # Pooling over the sequence dimension
        elif pooling == "max":
            X = X.max(dim=1).values  # Pooling over the sequence dimension
        
        # Convert PyTorch tensors to NumPy arrays
        X = X.numpy()  # Convert X to NumPy array
        y = y.numpy()  # Convert y to NumPy array

        # Append to lists
        X_list.append(X)
        y_list.append(y)
    
    # Concatenate all batches
    X_full = np.concatenate(X_list, axis=0)
    y_full = np.concatenate(y_list, axis=0)
    
    return X_full, y_full

def load_data(data, batch_sizes=(30, 30, 20), model_type="dl"):
    ### load embeddings ###
    with open("data/text2embeddings.pkl", "rb") as f:
        text2embeddings = pickle.load(f)

    train_bs, val_bs, test_bs = batch_sizes
    split_ratios = [0.7, 0.15, 0.15]

    random.seed(42)
    random.shuffle(data)
    train_data, val_data, test_data = split_list(data, split_ratios)

    if model_type == "dl":
        print(f"""
        train: {len(train_data)}
        val: {len(val_data)}
        test: {len(test_data)}
        """)

        ### setup dataloaders ###
        train_ds = PatientDataset(train_data, text2embeddings)
        val_ds = PatientDataset(val_data, text2embeddings, undersample=False)
        test_ds = PatientDataset(test_data, text2embeddings, undersample=False)

        train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=val_bs, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=test_bs, collate_fn=collate_fn, num_workers=5)

        return train_dl, val_dl, test_dl, (train_data, val_data, test_data)
    elif model_type == "ml-optuna":
        print(f"""
        train: {len(train_data)}
        val: {len(val_data)}
        test: {len(test_data)}
        """)

        ### setup dataloaders ###
        train_ds = PatientDataset(train_data, text2embeddings)
        val_ds = PatientDataset(val_data, text2embeddings, undersample=False)
        test_ds = PatientDataset(test_data, text2embeddings, undersample=False)

        train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=val_bs, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=test_bs, collate_fn=collate_fn, num_workers=5)

        train_X, train_y = dataloader_to_numpy(train_dl, pooling="mean")
        val_X, val_y = dataloader_to_numpy(val_dl, pooling="mean")
        test_X, test_y = dataloader_to_numpy(test_dl, pooling="mean")
        return train_X, train_y, val_X, val_y, test_X, test_y, (train_data, val_data, test_data)
    else:
        train_data += val_data
        print(f"""
        train: {len(train_data)}
        test: {len(test_data)}
        """)

        ### setup dataloaders ###
        train_ds = PatientDataset(train_data, text2embeddings)
        test_ds = PatientDataset(test_data, text2embeddings, undersample=False)

        train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=test_bs, collate_fn=collate_fn, num_workers=5)

        train_X, train_y = dataloader_to_numpy(train_dl, pooling="mean")
        test_X, test_y = dataloader_to_numpy(test_dl, pooling="mean")
        return train_X, train_y, test_X, test_y, (train_data, test_data)

def extract_features(dataloader, encoder, device="cuda"):
    features = []
    labels = []

    # Ensure the encoder is on the correct device and switch to evaluation mode
    encoder = encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data["embeddings"].to(device), data["label"]  # Move inputs to GPU
            feature = encoder(inputs)
            features.append(feature.cpu().numpy())  # Move features back to CPU and convert to numpy
            labels.append(targets.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels

class FC(L.LightningModule):
    def __init__(self, embedding_dim=768, hidden_dim=128, output_dim=1, batch_sizes=(30,30,20), agg_type="logits_max"):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = F1Score(task="binary")
        self.val_acc = F1Score(task="binary")
        self.test_acc = F1Score(task="binary")

        self.train_bs, self.val_bs, self.test_bs = batch_sizes

        self.agg_type = agg_type

    def forward(self, x):
        if self.agg_type == "sequence_mean":
            x = torch.mean(x, dim=1)  # average over the sequence length

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation - BCEWithLogitsLoss applies sigmoid
        return x

    def _common_step(self, batch, batch_idx, acc_metric):
        x, y = batch['embeddings'], batch['label']
        logits = self(x).squeeze(-1)  # remove the last singleton dimension if it exists

        if self.agg_type == "logits_mean":
            logits = logits.mean(dim=1)  # mean aggregation
        elif self.agg_type == "logits_max":
            logits = logits.max(dim=1).values  # max aggregation

        loss = self.loss_fn(logits, y.type_as(logits))
        acc_metric.update(logits.sigmoid(), y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.train_acc)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.train_bs)
        self.log('train_acc', self.train_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.train_bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.val_acc)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.val_bs)
        self.log('val_acc', self.val_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.val_bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.test_acc)
        self.log('test_loss', loss, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.test_bs)
        self.log('test_acc', self.test_acc.compute(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.test_bs)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class AttentionModelV2(L.LightningModule):
    def __init__(self, embedding_dim=768, hidden_dim=128, output_dim=1, num_heads=6, batch_size=(30,30,20)):
        super(AttentionModelV2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = F1Score(task="binary")
        self.val_acc = F1Score(task="binary")
        self.test_acc = F1Score(task="binary")

        self.train_bs, self.val_bs, self.test_bs = batch_size

    def forward(self, x):
        # x shape expected: [batch_size, seq_len, embedding_dim]
        x = x.permute(1, 0, 2)  # Now x is [seq_len, batch_size, embedding_dim]
        
        # apply attention
        query = x.mean(dim=0, keepdim=True)  # Reduce over sequence length, keep batch dimension
        attended_features, _ = self.attention_layer(query, x, x)
        
        # Reverting dimensions to match the linear layers
        attended_features = attended_features.permute(1, 0, 2)  # Back to [batch_size, seq_len, embedding_dim]
        # Optionally, flatten or pool the sequence dimension before passing to fully connected layers
        attended_features = attended_features.mean(dim=1)

        x = F.relu(self.fc1(attended_features))
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx, acc_metric):
        x, y = batch['embeddings'], batch['label']
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.type_as(logits))
        acc_metric.update(logits.sigmoid(), y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.train_acc)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.train_bs)
        self.log('train_acc', self.train_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.train_bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.val_acc)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.val_bs)
        self.log('val_acc', self.val_acc.compute(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.val_bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, self.test_acc)
        self.log('test_loss', loss, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.test_bs)
        self.log('test_acc', self.test_acc.compute(), on_step=True, on_epoch=True, sync_dist=True, batch_size=self.test_bs)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer