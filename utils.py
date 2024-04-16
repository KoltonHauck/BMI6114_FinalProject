import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import json

import torch.nn as nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import F1Score, Accuracy

import lightning as L
import random
import pickle

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from collections import Counter

from utils import *

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

        for encounter in patient['encounters']:
            # Fetch embeddings for each text piece
            for key in ['Description', 'ReasonDescription']:
                text = encounter['encounter'].get(key, '')
                if text and text in self.text2embedding:
                    embeddings.append(self.text2embedding[text])

            for item_type in ['conditions', 'careplans', 'procedures']:
                for item in encounter[item_type]:
                    for key in ['Description', 'ReasonDescription']:
                        text = item.get(key, '')
                        if text and text in self.text2embedding:
                            embeddings.append(self.text2embedding[text])

        # Stack embeddings if not empty, else return a zero tensor
        embeddings_tensor = torch.stack(embeddings) if embeddings else torch.zeros(1, len(next(iter(self.text2embedding.values()))))
    
        return {
            'embeddings': embeddings_tensor,  # a list of tensors
            'features': torch.tensor([patient['lat'], patient['lon']]),
            'label': int(patient["label"])
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
    
def collate_fn(batch):
    embeddings = [item['embeddings'] for item in batch]
    features = torch.stack([item['features'] for item in batch])
    labels = torch.tensor([int(item['label']) for item in batch])

    # Pad the embeddings sequences
    embeddings_padded = pad_sequence(embeddings, batch_first=True)

    return {
        'embeddings': embeddings_padded,
        'features': features,
        'label': labels
    }

class EncounterAutoencoder(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128):
        super(EncounterAutoencoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=20, batch_first=True)
        # self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return hidden[-1]  # Taking the last hidden state as the representation

        attn_output, _ = self.attention(output, output, output)
        return attn_output.mean(dim=1)  # Aggregate attention output

class PatientAutoencoder(L.LightningModule):
    def __init__(self, embedding_dim=768, hidden_dim=128, patient_latent_dim=64):
        super(PatientAutoencoder, self).__init__()
        self.save_hyperparameters()
        self.encounter_autoencoder = EncounterAutoencoder(embedding_dim, hidden_dim)
        self.patient_encoder = nn.Sequential(
            nn.Linear(hidden_dim, patient_latent_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.patient_decoder = nn.Sequential(
            nn.Linear(patient_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):

        encounter_representation = self.encounter_autoencoder(x['embeddings'])
        patient_encoded = self.patient_encoder(encounter_representation)
        patient_decoded = self.patient_decoder(patient_encoded)

        return patient_decoded
    
    def _common_step(self, batch, batch_idx, loss_metric, acc_metric, loss_lbl, metric_lbl):

        X = batch
        logits = self.forward(X)

        loss = loss_metric(logits, self.encounter_autoencoder(X["embeddings"]))

        self.log(loss_lbl, loss, prog_bar=True, sync_dist=True)
        self.log(metric_lbl, acc_metric(logits, self.encounter_autoencoder(X["embeddings"])), prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.train_mse, self.train_mae, "train_mse", "train_mae")

    def on_training_epoch_end(self):
        self.log("train_mse", self.train_mse.compute(), sync_dist=True)
        self.log("train_mae", self.train_mae.compute(), sync_dist=True)
        self.train_mse.reset()
        self.train_mae.reset()

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.val_mse, self.val_mae, "val_mse", "val_mae")

    def on_validation_epoch_end(self):
        self.log("val_mse", self.val_mse.compute(), sync_dist=True)
        self.log("val_mae", self.val_mae.compute(), sync_dist=True)
        self.val_mse.reset()
        self.val_mae.reset()

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.val_mse, self.val_mae, "test_mse", "test_mae")

    def on_test_epoch_end(self):
        self.log("test_mse", self.test_mse.compute(), sync_dist=True)
        self.log("test_mae", self.test_mae.compute(), sync_dist=True)
        self.test_mse.reset()
        self.test_mae.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

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

class PatientClassifier(L.LightningModule):
    def __init__(self, encoder, hidden_dim):
        super().__init__()

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(hidden_dim, 1)

        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_acc = Accuracy(task="binary")
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x):
        encoded_features = self.encoder(x)
        logits = self.classifier(encoded_features)
        return logits

    def _common_step(self, batch, batch_idx, metric, loss_lbl, metric_lbl):
        x, y = batch["embeddings"], batch["label"]

        logits = self.forward(x)
        loss = self.loss_function(logits.view(-1), y.float())  # Ensure y is float and logits are reshaped appropriately

        preds = torch.sigmoid(logits) > 0.5  # Calculate predictions based on the sigmoid threshold
        metric(preds.view(-1), y)  # Update metric
        
        self.log(loss_lbl, loss, prog_bar=True)
        self.log(metric_lbl, metric.compute(), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.train_acc, "train_loss", "train_acc")

    def on_training_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.val_acc, "val_loss", "val_acc")

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, self.test_acc, "test_loss", "test_acc")

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.0001)
        return optimizer

