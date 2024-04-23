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
import numpy as np

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
            'label': int(patient["label"]),
            'patient_id': patient['id']
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

class PatientDatasetV2(Dataset):
    def __init__(self, data, text2embedding, undersample=True):
        self.text2embedding = text2embedding
        self.embedding_dim = len(next(iter(text2embedding.values())))  # Assuming all embeddings have the same length
        if undersample:
            self.data = self.undersample_data(data)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        patient_embeddings = {'encounters': []}

        for encounter in patient['encounters']:
            encounter_embedding = {'encounter': {}, 'conditions': [], 'careplans': [], 'procedures': []}

            # Fetch embeddings for encounter
            for key in ['Description', 'ReasonDescription']:
                text = encounter['encounter'].get(key, '')
                encounter_embedding['encounter'][key] = self.get_embedding(text)

            # Fetch embeddings for items
            for item_type in ['conditions', 'careplans', 'procedures']:
                for item in encounter[item_type]:
                    item_embeddings = {}
                    for key in ['Description', 'ReasonDescription']:
                        text = item.get(key, '')
                        item_embeddings[key] = self.get_embedding(text)
                    encounter_embedding[item_type].append(item_embeddings)

            patient_embeddings['encounters'].append(encounter_embedding)

        return {
            'embeddings': patient_embeddings['encounters'],
            'features': torch.tensor([patient['lat'], patient['lon']]),
            'label': int(patient["label"])
        }

    def get_embedding(self, text):
        if text and text in self.text2embedding:
            return self.text2embedding[text]
        else:
            return torch.zeros(self.embedding_dim)  # Return zero vector if not found

    def undersample_data(self, data):
        label_counts = Counter(patient["label"] for patient in data)
        min_count = min(label_counts.values())
        undersampled_data = []
        counts = {label: 0 for label in label_counts}
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

def collate_fnV2(batch):
    embeddings = [item['embeddings'] for item in batch]
    features = torch.stack([item['features'] for item in batch])
    labels = torch.tensor([int(item['label']) for item in batch])

    # Pad embeddings manually for nested structure
    embeddings_padded = custom_pad(embeddings)

    return {
        'embeddings': embeddings_padded,
        'features': features,
        'label': labels
    }

def collate_fn(batch):
    embeddings = [item['embeddings'] for item in batch]
    features = torch.stack([item['features'] for item in batch])
    labels = torch.tensor([int(item['label']) for item in batch])
    ids = [item['id'] for item in batch]

    # Pad the embeddings sequences
    embeddings_padded = pad_sequence(embeddings, batch_first=True)

    return {
        'embeddings': embeddings_padded,
        'features': features,
        'label': labels,
        'patient_id': ids
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

class EncounterAutoencoderV2(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, output_dim=128):
        super(EncounterAutoencoderV2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define LSTMs for different parts of an encounter
        self.encounter_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.conditions_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True)
        self.careplans_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True)
        self.procedures_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True)

        # A linear layer to merge all different LSTM outputs
        self.fc = nn.Linear(hidden_dim + 3 * (hidden_dim // 2), output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Assuming x is a dict with structure given by the new dataset format
        encounter_out, _ = self.encounter_lstm(x['encounter'])
        conditions_out, _ = self.conditions_lstm(torch.cat(x['conditions'], dim=0))
        careplans_out, _ = self.careplans_lstm(torch.cat(x['careplans'], dim=0))
        procedures_out, _ = self.procedures_lstm(torch.cat(x['procedures'], dim=0))

        # Taking the last hidden state as the representation
        encounter_rep = encounter_out[:, -1, :]
        conditions_rep = conditions_out[:, -1, :]
        careplans_rep = careplans_out[:, -1, :]
        procedures_rep = procedures_out[:, -1, :]

        # Concatenate all outputs
        concatenated = torch.cat((encounter_rep, conditions_rep, careplans_rep, procedures_rep), dim=1)
        
        # Pass through a final linear layer
        final_output = self.fc(concatenated)
        final_output = self.relu(final_output)

        return final_output

class PatientAutoencoderV2(L.LightningModule):
    def __init__(self, embedding_dim=768, hidden_dim=128, patient_latent_dim=64, output_dim=128):
        super(PatientAutoencoderV2, self).__init__()
        self.save_hyperparameters()
        self.encounter_autoencoder = EncounterAutoencoderV2(embedding_dim, hidden_dim, output_dim)
        self.patient_encoder = nn.Sequential(
            nn.Linear(output_dim, patient_latent_dim),  # Adjusted to match EncounterAutoencoder's output
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.patient_decoder = nn.Sequential(
            nn.Linear(patient_latent_dim, output_dim),  # Ensure the output dimension matches the encoder's input
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.train_mse = nn.MSELoss()
        self.val_mse = nn.MSELoss()
        self.test_mse = nn.MSELoss()

        self.train_mae = nn.L1Loss()
        self.val_mae = nn.L1Loss()
        self.test_mae = nn.L1Loss()

    def forward(self, x):
        encounter_representation = self.encounter_autoencoder(x['embeddings'])
        patient_encoded = self.patient_encoder(encounter_representation)
        patient_decoded = self.patient_decoder(patient_encoded)
        return patient_decoded

    # Adjust the common step to match the new structure
    def _common_step(self, batch, batch_idx, loss_metric, acc_metric, loss_lbl, metric_lbl):
        embeddings = batch['embeddings']
        logits = self.forward(embeddings)
        labels = self.encounter_autoencoder(embeddings)

        loss = loss_metric(logits, labels)
        self.log(loss_lbl, loss, prog_bar=True)
        self.log(metric_lbl, acc_metric(logits, labels), prog_bar=True)
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

def dataloader_to_numpy(dataloader, pooling='mean'):
    """Convert a PyTorch DataLoader to NumPy arrays with optional pooling."""
    X_list, y_list = [], []
    
    # Loop over the DataLoader
    for batch in dataloader:
        X, y = batch['embeddings'], batch['label']
        
        # Check the desired pooling method
        if pooling == 'mean':
            X = X.mean(dim=1)  # Pooling over the sequence dimension
        elif pooling == 'max':
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

def load_data(data, batch_sizes=(20, 20, 20), model_type="dl"):
    ### load embeddings ###
    with open('data/text2embeddings.pkl', 'rb') as f:
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

        train_X, train_y = dataloader_to_numpy(train_dl, pooling='mean')
        val_X, val_y = dataloader_to_numpy(val_dl, pooling='mean')
        test_X, test_y = dataloader_to_numpy(test_dl, pooling='mean')
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

        train_X, train_y = dataloader_to_numpy(train_dl, pooling='mean')
        test_X, test_y = dataloader_to_numpy(test_dl, pooling='mean')
        return train_X, train_y, test_X, test_y, (train_data, test_data)



def load_dataV2(data, batch_sizes=(20, 20, 20)):
    train_bs, val_bs, test_bs = batch_sizes
    split_ratios = [0.7, 0.15, 0.15]

    random.shuffle(data)
    train_data, val_data, test_data = split_list(data, split_ratios)

    print(f"""
    train: {len(train_data)}
    val: {len(val_data)}
    test: {len(test_data)}
    """)

    ### load embeddings ###
    with open('data/text2embeddings.pkl', 'rb') as f:
        text2embeddings = pickle.load(f)

    ### setup dataloaders ###
    train_ds = PatientDatasetV2(train_data, text2embeddings)
    val_ds = PatientDatasetV2(val_data, text2embeddings, undersample=False)
    test_ds = PatientDatasetV2(test_data, text2embeddings, undersample=False)

    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fnV2, num_workers=5, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=val_bs, collate_fn=collate_fnV2, num_workers=5, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=test_bs, collate_fn=collate_fnV2, num_workers=5)

    return train_dl, val_dl, test_dl, (train_data, val_data, test_data)

def extract_features(dataloader, encoder, device='cuda'):
    features = []
    labels = []

    # Ensure the encoder is on the correct device and switch to evaluation mode
    encoder = encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data['embeddings'].to(device), data['label']  # Move inputs to GPU
            feature = encoder(inputs)
            features.append(feature.cpu().numpy())  # Move features back to CPU and convert to numpy
            labels.append(targets.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels
