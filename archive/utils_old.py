class Attention(nn.Module):
    def __init__(self, feature_size):
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(feature_size, feature_size)
        self.key_projection = nn.Linear(feature_size, feature_size)
        self.value_projection = nn.Linear(feature_size, feature_size)

    def forward(self, query, keys, values):
        query_proj = self.query_projection(query)
        keys_proj = self.key_projection(keys)
        values_proj = self.value_projection(values)

        scores = torch.matmul(query_proj, keys_proj.transpose(-2, -1)) / (keys_proj.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(weights, values_proj)
        return attended.squeeze(1), weights  # assume the query is batch-wise
    
class AttentionModel(L.LightningModule):
    def __init__(self, embedding_dim=768, hidden_dim=128, output_dim=1):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.attention_layer = Attention(embedding_dim)
        # self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        # Assume x is [batch_size, seq_len, embedding_dim]
        # Apply attention
        query = x.mean(dim=1, keepdim=True)  # A simple way to obtain a query from inputs
        attended_features, _ = self.attention_layer(query, x, x)

        x = F.relu(self.fc1(attended_features))
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch['embeddings'], batch['label']
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.type_as(logits))
        self.accuracy.update(logits.sigmoid(), y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        self.log('train_acc', self.accuracy.compute(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        self.log('val_acc', self.accuracy.compute(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True, on_step=True, on_epoch=False)
        self.log('test_acc', self.accuracy.compute(), on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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
    with open("data/text2embeddings.pkl", "rb") as f:
        text2embeddings = pickle.load(f)

    ### setup dataloaders ###
    train_ds = PatientDatasetV2(train_data, text2embeddings)
    val_ds = PatientDatasetV2(val_data, text2embeddings, undersample=False)
    test_ds = PatientDatasetV2(test_data, text2embeddings, undersample=False)

    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fnV2, num_workers=5, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=val_bs, collate_fn=collate_fnV2, num_workers=5, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=test_bs, collate_fn=collate_fnV2, num_workers=5)

    return train_dl, val_dl, test_dl, (train_data, val_data, test_data)

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
        encounter_out, _ = self.encounter_lstm(x["encounter"])
        conditions_out, _ = self.conditions_lstm(torch.cat(x["conditions"], dim=0))
        careplans_out, _ = self.careplans_lstm(torch.cat(x["careplans"], dim=0))
        procedures_out, _ = self.procedures_lstm(torch.cat(x["procedures"], dim=0))

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
            nn.Linear(output_dim, patient_latent_dim),  # Adjusted to match EncounterAutoencoder"s output
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.patient_decoder = nn.Sequential(
            nn.Linear(patient_latent_dim, output_dim),  # Ensure the output dimension matches the encoder"s input
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
        encounter_representation = self.encounter_autoencoder(x["embeddings"])
        patient_encoded = self.patient_encoder(encounter_representation)
        patient_decoded = self.patient_decoder(patient_encoded)
        return patient_decoded

    # Adjust the common step to match the new structure
    def _common_step(self, batch, batch_idx, loss_metric, acc_metric, loss_lbl, metric_lbl):
        embeddings = batch["embeddings"]
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

        encounter_representation = self.encounter_autoencoder(x["embeddings"])
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

def collate_fnV2(batch):
    embeddings = [item["embeddings"] for item in batch]
    features = torch.stack([item["features"] for item in batch])
    labels = torch.tensor([int(item["label"]) for item in batch])

    # Pad embeddings manually for nested structure
    embeddings_padded = custom_pad(embeddings)

    return {
        "embeddings": embeddings_padded,
        "features": features,
        "label": labels
    }

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
        patient_embeddings = {"encounters": []}

        for encounter in patient["encounters"]:
            encounter_embedding = {"encounter": {}, "conditions": [], "careplans": [], "procedures": []}

            # Fetch embeddings for encounter
            for key in ["Description", "ReasonDescription"]:
                text = encounter["encounter"].get(key, "")
                encounter_embedding["encounter"][key] = self.get_embedding(text)

            # Fetch embeddings for items
            for item_type in ["conditions", "careplans", "procedures"]:
                for item in encounter[item_type]:
                    item_embeddings = {}
                    for key in ["Description", "ReasonDescription"]:
                        text = item.get(key, "")
                        item_embeddings[key] = self.get_embedding(text)
                    encounter_embedding[item_type].append(item_embeddings)

            patient_embeddings["encounters"].append(encounter_embedding)

        return {
            "embeddings": patient_embeddings["encounters"],
            "features": torch.tensor([patient["lat"], patient["lon"]]),
            "label": int(patient["label"])
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

def embed_descriptions(desc2code, model_directory, batch_size=32):
    """ Given a dictionary of SNOMED code:description k:v pairs, and an embedding model (IE BERT), returns a dict of description:embedding of each code"""
    tokenizer = BertTokenizer.from_pretrained(model_directory)

    descriptions = list(desc2code.keys())

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(model_directory)
    model.eval()  # Put the model in "evaluation" mode, which turns off dropout
    print(f"model loaded | generating embeddings with bs={batch_size}")

    inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=64)
    # Assuming you have enough memory, process the entire list in batches
    embeddings = []

    # Process in batches with tqdm for progress tracking
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating Embeddings"):
        batch = inputs[i:i+batch_size]
        with torch.no_grad():
            outputs = model(**batch)

        # Extract pooled output embeddings
        batch_embeddings = outputs.pooler_output
        embeddings.append(batch_embeddings)

    # Concatenate batched embeddings
    embeddings = torch.cat(embeddings, dim=0)

    return {descriptions[i]: embeddings[i] for i in range(len(descriptions))}

def get_code2desc_dict(path_to_synthea_folder, synthea_table_information):
    """ Given path to folder containing synthea dataset, and synthea_table_information_dict, return dictionary of all SNOMED code:description pairs"""
    code2description = {}
    description2code = {}

    for table_name, table_data in synthea_table_information.items():
        print(f"processing {table_name}")
        df = pd.read_csv(os.path.join(path_to_synthea_folder, table_name), header=None, dtype=str)
        df.columns = table_data["columns"]

        for pair in table_data.get("code_description_pairs", []):
            tmp_df = df[pair].dropna()

            tmp_df[pair[1]] = tmp_df[pair[1]].apply(remove_text_inside_parentheses)

            if tmp_df.empty:
                continue

            description2code.update(pd.Series(tmp_df[pair[0]].values, index=tmp_df[pair[1]]).to_dict())
            code2description.update(pd.Series(tmp_df[pair[1]].values, index=tmp_df[pair[0]]).to_dict())

    return code2description, description2code

def get_average_embeddings(df, description_columns, desc2embeddings):
    embeddings = {}
    
    for col in description_columns:
        tmp_emb = [desc2embeddings[desc] for desc in df[col] if desc != ""]
        if tmp_emb:
            emb_torch = torch.stack(tmp_emb)
            flattened_emb = emb_torch.mean(dim=0)
            embeddings[col] = flattened_emb
        else:
            embeddings[col] = torch.zeros(768)

    return embeddings

class GRUEncoderWithAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GRUEncoderWithAttention, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask=None):  # Add mask as an optional argument
        # x is a batch of sequences of embeddings: shape (batch_size, seq_length, embedding_dim)
        gru_outputs, _ = self.gru(x)  # gru_outputs shape: (batch_size, seq_length, hidden_dim)
        
        # Apply attention, passing the mask to the attention mechanism
        context_vector, attention_weights = self.attention(gru_outputs, mask)
        return context_vector, attention_weights

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, gru_outputs, mask=None):
        # gru_outputs shape: [batch_size, seq_length, hidden_dim]
        energies = self.linear(gru_outputs).squeeze(-1)  # [batch_size, seq_length]

        if mask is not None:
            energies = energies.masked_fill(mask == 0, float('-inf'))  # Masking with -inf before softmax

        attention_weights = nn.functional.softmax(energies, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), gru_outputs).squeeze(1)

        return context_vector, attention_weights

def vectorize_encounters(encounters_df, procedures_df, desc2embeddings, synthea_table_information):
    """ Function to vectorize patient information """
    
    enc_vectors = []

    encounter_ids = encounters_df.Id.to_list()
    for encounter_id in encounter_ids: # iter over encounters
        curr_encounter_vector = []

        # add vectorized encounter data
        curr_encounter = encounters_df[encounters_df.Id == encounter_id]
        encounter_embeddings = get_average_embeddings(curr_encounter, synthea_table_information["encounters.csv"]["description_columns"], desc2embeddings)
        for emb in encounter_embeddings.values():
            curr_encounter_vector.extend(emb)

        # add vectorized procedures
        encounter_procedures = procedures_df[procedures_df["Encounter"] == encounter_id]
        procedure_embeddings = get_average_embeddings(encounter_procedures, synthea_table_information["procedures.csv"]["description_columns"], desc2embeddings)
        for emb in procedure_embeddings.values():
            curr_encounter_vector.extend(emb)

        enc_vectors.append(torch.tensor(curr_encounter_vector))

    return torch.stack(enc_vectors, dim=0)
