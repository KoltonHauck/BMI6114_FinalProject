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

from utils import *

def main():
    with open("data/all_v2.json","r") as j_file:
        data = json.load(j_file)

    split_ratios = [0.7, 0.15, 0.15]

    random.shuffle(data)
    train_data, val_data, test_data = split_list(data, split_ratios)

    print(f"""
    train: {len(train_data)}
    val: {len(val_data)}
    test: {len(test_data)}
    """)


    with open('data/text2embeddings.pkl', 'rb') as f:
        text2embeddings = pickle.load(f)

    embedding_dim = 768
    hidden_dim = 128  # Intermediate representation size
    patient_latent_dim = 64  # Final latent space dimension for the patient

    pretrained_hae = PatientAutoencoder.load_from_checkpoint(checkpoint_path='checkpoints/last.ckpt')
    encoder = nn.Sequential(
        pretrained_hae.encounter_autoencoder,
        pretrained_hae.patient_encoder
    )

    classifier_model = PatientClassifier(encoder=encoder, hidden_dim=patient_latent_dim)

    train_ds, val_ds, test_ds = PatientDataset(train_data, text2embeddings), PatientDataset(val_data, text2embeddings), PatientDataset(test_data, text2embeddings)
    train_dl = DataLoader(train_ds, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=20, collate_fn=collate_fn, num_workers=5, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=20, collate_fn=collate_fn, num_workers=5)

    logger = CSVLogger("pl_logs", name="hae_classifier")

    # Initialize PyTorch Lightning trainer and train the model
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu",
        strategy="ddp_spawn",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_last=True, save_top_k=1,
                dirpath="checkpoints/", filename="hae_classifier-{epoch:02d}-{val_loss:.2f}"
            ),
            EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ],
        logger=logger,
        devices=2,
        log_every_n_steps=50
    )

    trainer.fit(classifier_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == "__main__":
    main()