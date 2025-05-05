# Copyright (C) 2025 CRCODE22
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.0
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import gradio as gr
import plotly.express as px
import time
from datetime import datetime, timedelta
import os
import pickle

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define the Identity Generator Neural Network
class IdentityGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IdentityGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to generate synthetic dataset with reordered columns (excluding Clone Number)
def generate_synthetic_dataset(num_samples=10000):
    data = {
        'Firstname': np.random.choice(['Emma', 'Olivia', 'Ava', 'Sophia', 'Amelia', 'Charlotte', 'Kisa', 'Lilith', 'Susan', 'Joyce', 'Zara', 'Luna', 'Nova', 'Aria', 'Mila'], num_samples),
        'Lastname': np.random.choice(['Smith', 'Johnson', 'Brown', 'Taylor', 'Wilson', 'Gadot', 'Roovers', 'Anderson', 'Lee', 'Kim', 'Patel', 'Garcia', 'Nguyen'], num_samples),
        'Nickname': np.random.choice(['Em', 'Liv', 'Ava', 'Sophie', 'Amy', 'Charlie', 'Bella', 'Zee', 'Luna', 'Nova'], num_samples),
        'Age': np.random.uniform(18, 40, num_samples),
        'Born': pd.to_datetime([datetime.now() - timedelta(days=np.random.uniform(18, 40) * 365) for _ in range(num_samples)]).strftime('%Y-%m-%d'),
        'Nationality': np.random.choice(['American', 'British', 'French', 'Japanese', 'Australian', 'Dutch', 'Russian', 'Ukrainian', 'Indian', 'Brazilian', 'Chinese'], num_samples),
        'Ethnicity': np.random.choice(['Caucasian', 'Asian', 'African', 'Mixed', 'Latina', 'South Asian', 'Middle Eastern'], num_samples),
        'Birthplace': np.random.choice(['New York, USA', 'London, UK', 'Paris, France', 'Tokyo, Japan', 'Netherlands', 'Russia', 'China', 'Turkey', 'Ukraine', 'Spain', 'Italy', 'Brazil', 'Iran', 'India', 'Australia'], num_samples),
        'Profession': np.random.choice(['Doctor', 'Engineer', 'Artist', 'Teacher', 'Designer', 'Hacker', 'Pornstar', 'Model', 'Actress', 'Scientist', 'Nurse', 'Pilot', 'Chef', 'Writer', 'Musician'], num_samples),
        'Height': np.random.uniform(150, 180, num_samples),
        'Weight': np.random.uniform(45, 80, num_samples),
        'Body type': np.random.choice(['Slim', 'Athletic', 'Curvy', 'Petite', 'Skinny', 'Average'], num_samples),
        'Body Measurements': [f"{np.random.randint(80, 100)}-{np.random.randint(55, 65)}-{np.random.randint(85, 105)}" for _ in range(num_samples)],
        'Hair color': np.random.choice(['Blonde', 'Brown', 'Black', 'Red', 'Purple', 'Green', 'Blue', 'Silver'], num_samples),
        'Eye color': np.random.choice(['Blue', 'Green', 'Brown', 'Hazel', 'Gray', 'Amber'], num_samples),
        'Bra/cup size': np.random.choice(['A', 'B', 'C', 'D', 'DD'], num_samples),
        'Boobs': np.random.choice(['Natural', 'Enhanced'], num_samples),
    }
    df = pd.DataFrame(data)
    df['Born'] = pd.to_datetime(df['Age'].apply(lambda age: datetime.now() - timedelta(days=age * 365)))
    df['Born'] = df['Born'].dt.strftime('%Y-%m-%d')

    numerical_columns = ['Age', 'Height', 'Weight']
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].mean())
        if np.any(np.isinf(df[col].astype(float))):
            df[col] = df[col].replace([np.inf, -np.inf], df[col].mean())

    return df

# Function to load dataset from dataset.csv or generate synthetic dataset
def load_or_generate_dataset(num_samples=10000):
    dataset_path = "dataset.csv"
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        # Ensure all required columns are present
        required_columns = ['Firstname', 'Lastname', 'Nickname', 'Age', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 
                            'Profession', 'Height', 'Weight', 'Body type', 'Body Measurements', 'Hair color', 'Eye color', 
                            'Bra/cup size', 'Boobs']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"dataset.csv is missing required columns: {missing_columns}")
        return df
    else:
        print(f"No dataset.csv found, generating synthetic dataset with {num_samples} samples")
        return generate_synthetic_dataset(num_samples)

# Preprocess the dataset with reordered columns
def preprocess_data(df):
    numerical_columns = ['Age', 'Height', 'Weight']
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
        df[col] = df[col].clip(lower=df[col].mean() - 3*df[col].std(), upper=df[col].mean() + 3*df[col].std())

    scaler_age = MinMaxScaler(feature_range=(0, 1))
    scaler_height = MinMaxScaler(feature_range=(0, 1))
    scaler_weight = MinMaxScaler(feature_range=(0, 1))
    scaler_measurements = MinMaxScaler(feature_range=(0, 1))

    df['Age'] = scaler_age.fit_transform(df[['Age']])
    df['Height'] = scaler_height.fit_transform(df[['Height']])
    df['Weight'] = scaler_weight.fit_transform(df[['Weight']])

    le_dict = {}
    categorical_columns = ['Firstname', 'Lastname', 'Nickname', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    measurements = np.array([list(map(int, m.split('-'))) for m in df['Body Measurements']])
    measurements_scaled = scaler_measurements.fit_transform(measurements)
    df['Body Measurements'] = measurements_scaled.tolist()

    return df, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements

# Convert data to tensor with reordered columns
def df_to_tensor(df):
    categorical_columns = ['Firstname', 'Lastname', 'Nickname', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']
    numerical_columns = ['Age', 'Height', 'Weight']
    features = df[categorical_columns + numerical_columns].values
    measurements = np.array(df['Body Measurements'].tolist())
    features = np.hstack((features, measurements))

    scaler_features = MinMaxScaler()
    features = np.nan_to_num(features, nan=np.nanmean(features), posinf=np.nanmean(features), neginf=np.nanmean(features))
    features = scaler_features.fit_transform(features)

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Features contain NaN or infinite values")

    return torch.FloatTensor(features), scaler_features

# Train the model with self-improvement cycles and console logs
def train_model(resume=False, le_dict=None, scaler_age=None, scaler_height=None, scaler_weight=None, scaler_measurements=None, scaler_features=None, df=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 19  # 13 categorical + 3 numerical + 3 measurements
    hidden_dim = 256  # Increased hidden_dim to allow for more complex patterns
    output_dim = 19
    model = IdentityGenerator(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    num_cycles = 10  # Reduced cycles since the model converges quickly
    epochs_per_cycle = 50
    batch_size = 64
    cycle_progress_range = 100 / num_cycles

    early_stopping_patience = 20
    min_delta = 1e-4  # Increased min_delta to allow for smaller improvements
    best_loss = float('inf')
    patience_counter = 0
    last_lr = optimizer.param_groups[0]['lr']
    total_epochs_completed = 0

    if resume and os.path.exists("checkpoint_model.pth"):
        try:
            checkpoint = torch.load("checkpoint_model.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            df = pd.read_csv("checkpoint_dataset.csv")
            with open("checkpoint_encoders.pkl", "rb") as f:
                checkpoint_encoders = pickle.load(f)
                le_dict = checkpoint_encoders['le_dict']
                scaler_age = checkpoint_encoders['scaler_age']
                scaler_height = checkpoint_encoders['scaler_height']
                scaler_weight = checkpoint_encoders['scaler_weight']
                scaler_measurements = checkpoint_encoders['scaler_measurements']
                scaler_features = checkpoint_encoders['scaler_features']
            yield None, None, 0, "Resumed from checkpoint", []
            yield model, device, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features
            return
        except Exception as e:
            yield None, None, 0, f"Failed to resume: {e}", []
            return

    data_tensor, scaler_features = df_to_tensor(df)
    data_tensor = data_tensor.to(device)

    training_start_time = time.time()  # Start timing the entire training process
    loss_history = []
    for cycle in range(num_cycles):
        cycle_start_progress = cycle * cycle_progress_range
        print(f"Starting Cycle {cycle + 1}/{num_cycles}")
        yield None, None, cycle_start_progress, f"Starting Cycle {cycle + 1}", loss_history

        for epoch in range(epochs_per_cycle):
            epoch_start_time = time.time()
            model.train()
            num_batches = len(df) // batch_size
            epoch_loss = 0.0

            indices = torch.randperm(len(data_tensor))
            data_tensor = data_tensor[indices]

            # Add small noise to the input data to prevent overfitting
            noise = torch.randn_like(data_tensor) * 0.01
            noisy_data = data_tensor + noise

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                batch_data = noisy_data[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches

            # Early stopping with tolerance
            if avg_epoch_loss < best_loss - min_delta:
                best_loss = avg_epoch_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss
                }, "best_model.pth")
                df.to_csv("checkpoint_dataset.csv", index=False)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at Cycle {cycle + 1}, Epoch {epoch + 1}")
                    break

            scheduler.step(avg_epoch_loss)
            current_lr = scheduler.get_last_lr()[0]
            # Reset patience counter if learning rate changes
            if current_lr != last_lr:
                patience_counter = 0
                last_lr = current_lr
            print(f"Learning Rate: {current_lr:.6f}")

            loss_history.append(avg_epoch_loss)
            epoch_progress = (epoch + 1) / epochs_per_cycle * cycle_progress_range
            total_progress = cycle_start_progress + epoch_progress

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            speed = 1 / epoch_time if epoch_time > 0 else float('inf')
            print(f"Cycle {cycle + 1}/{num_cycles}, Epoch {epoch + 1}/{epochs_per_cycle} | "
                  f"Avg Loss: {avg_epoch_loss:.6f} | Time: {epoch_time:.2f}s | Speed: {speed:.2f} epochs/s")

            status = f"Training Model (Cycle {cycle + 1}, Epoch {epoch + 1}/{epochs_per_cycle}) - Avg Loss: {avg_epoch_loss:.6f}"
            yield None, None, total_progress, status, loss_history

            total_epochs_completed += 1

        if patience_counter >= early_stopping_patience:
            break  # Exit the cycle loop if early stopping was triggered

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Training Summary: Total Epochs: {total_epochs_completed}, Final Loss: {best_loss:.6f}, Total Time: {total_training_time:.2f}s")

    yield model, device, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features

# Generate a new identity with reordered columns
def generate_identity(model, device, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(1, 19).to(device)
        generated = model(noise).cpu().numpy()[0]

        generated_full = scaler_features.inverse_transform(generated.reshape(1, -1))[0]

        identity = {}
        categorical_columns = ['Firstname', 'Lastname', 'Nickname', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']

        for idx, col in enumerate(categorical_columns):
            value = int(round(generated_full[idx]))
            num_classes = len(le_dict[col].classes_)
            value = max(0, min(value, num_classes - 1))
            identity[col] = le_dict[col].inverse_transform([value])[0]

        identity['Age'] = float(scaler_age.inverse_transform([[generated_full[13]]])[0][0])
        identity['Height'] = float(scaler_height.inverse_transform([[generated_full[14]]])[0][0])
        identity['Weight'] = float(scaler_weight.inverse_transform([[generated_full[15]]])[0][0])

        measurements = generated_full[16:19]
        measurements = scaler_measurements.inverse_transform([measurements])[0]
        measurements = np.clip(measurements, [80, 55, 85], [100, 65, 105])
        identity['Body Measurements'] = f"{int(measurements[0])}-{int(measurements[1])}-{int(measurements[2])}"

        identity['Age'] = max(18, min(40, round(identity['Age'])))
        identity['Height'] = max(150, min(180, round(identity['Height'])))
        identity['Weight'] = max(45, min(80, round(identity['Weight'])))

        return identity

# Filter generated identities
def filter_identity(identity):
    try:
        if not (18 <= identity['Age'] <= 40):
            return False
        if not (150 <= identity['Height'] <= 180):
            return False
        if not (45 <= identity['Weight'] <= 80):
            return False

        measurements = identity['Body Measurements'].split('-')
        if len(measurements) != 3:
            return False

        required_fields = ['Hair color', 'Eye color', 'Body type', 'Bra/cup size', 'Boobs', 'Firstname', 'Lastname']
        for field in required_fields:
            if not identity.get(field):
                return False

        return True
    except Exception as e:
        return False

# Gradio interface function with reordered output, Clone Number, real-time graph, and progress bar
def generate_identities_gui(num_identities, resume_training, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features, df):
    yield None, None, 0, "Starting Training...", None

    train_gen = train_model(resume=resume_training, le_dict=le_dict, scaler_age=scaler_age, scaler_height=scaler_height, scaler_weight=scaler_weight, scaler_measurements=scaler_measurements, scaler_features=scaler_features, df=df)
    model = None
    device = None

    for train_output in train_gen:
        if isinstance(train_output, tuple) and len(train_output) == 8:
            model, device, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features = train_output
            yield None, None, 50, "Model Loaded - Starting Generation", None
            break
        _, _, progress, status, loss_history = train_output
        if loss_history:
            loss_df = pd.DataFrame({
                'Epoch': list(range(1, len(loss_history) + 1)),
                'Loss': loss_history
            })
            fig = px.line(loss_df, x='Epoch', y='Loss', title="Training Loss Over Time")
            fig.update_layout(
                plot_bgcolor='rgba(10, 10, 40, 0.8)',
                paper_bgcolor='rgba(10, 10, 40, 0.8)',
                font_color='#00ffcc',
                title_font_color='#00ffcc',
                xaxis=dict(gridcolor='rgba(0, 230, 230, 0.2)'),
                yaxis=dict(gridcolor='rgba(0, 230, 230, 0.2)')
            )
            loss_plot = fig
        else:
            loss_plot = None
        yield None, None, progress, status, loss_plot

    if model is None:
        yield None, None, 50, "Training failed - cannot generate identities", None
        return

    generation_start_time = time.time()
    identities = []
    num_identities = int(num_identities)
    for i in range(num_identities):
        progress_value = 50 + (i / num_identities) * 25
        status = f"Generating Identity {i+1}/{num_identities}"
        yield None, None, progress_value, status, None
        identity = generate_identity(model, device, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features)
        if identity:
            # Add Clone Number as a string
            identity['Clone Number'] = f"CLN-{str(i+1).zfill(3)}"
            identities.append(identity)

    generation_end_time = time.time()
    generation_time = generation_end_time - generation_start_time
    print(f"Generated {len(identities)} identities in {generation_time:.2f}s")

    filtering_start_time = time.time()
    filtered_identities = []
    for i, identity in enumerate(identities):
        progress_value = 75 + (i / len(identities)) * 25
        status = f"Filtering Identity {i+1}/{len(identities)}"
        yield None, None, progress_value, status, None
        if filter_identity(identity):
            filtered_identities.append(identity)

    filtering_end_time = time.time()
    filtering_time = filtering_end_time - filtering_start_time
    print(f"Filtered {len(filtered_identities)} identities (out of {len(identities)}) in {filtering_time:.2f}s")

    if filtered_identities:
        desired_columns = ['Clone Number', 'Firstname', 'Lastname', 'Nickname', 'Age', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Height', 'Weight', 'Body type', 'Body Measurements', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']
        df_identities = pd.DataFrame(filtered_identities)
        df_identities = df_identities[desired_columns]
        df_identities['Age'] = df_identities['Age'].round(0).astype(int)
        df_identities['Height'] = df_identities['Height'].round(0).astype(int)
        df_identities['Weight'] = df_identities['Weight'].round(0).astype(int)
        df_identities['Body Measurements'] = df_identities['Body Measurements'].apply(
            lambda x: '-'.join(str(int(float(v))) for v in x.split('-'))
        )
        csv_path = "generated_cha_identities.csv"
        df_identities.to_csv(csv_path, index=False)
        yield df_identities, csv_path, 100, "Generation Complete", None
    else:
        yield None, None, 100, "Generation Complete - No valid identities", None

# Wrapper for Gradio
def generate_identities_gui_wrapper(num_identities, resume_training):
    df = load_or_generate_dataset()
    df, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements = preprocess_data(df)
    data_tensor, scaler_features = df_to_tensor(df)
    yield from generate_identities_gui(num_identities, resume_training, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features, df)

# Simplified CSS for Matrix integration, including loss plot styling
custom_css = """
body {
    background: transparent;
    color: #00e6e6;
}
.gradio-container {
    max-width: 600px;
    margin: auto;
    border: 2px solid #00e6e6;
    border-radius: 15px;
    background: rgba(10, 10, 40, 0.8);
    padding: 20px;
}
h1 {
    text-align: center;
    color: #00ffcc;
    text-shadow: 0 0 15px #00ffcc;
}
h2, h3 {
    text-align: center;
    color: #00ffcc;
}
button {
    background: #1a1a4d;
    color: #00e6e6;
    border: 2px solid #00e6e6;
    border-radius: 10px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
button:hover {
    background: #00e6e6;
    color: #0d0d2b;
    box-shadow: 0 0 25px #00e6e6;
    transform: scale(1.1);
}
.dataframe-container {
    width: 100% !important;
    background: rgba(20, 20, 60, 0.9);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    padding: 10px;
}
.dataframe table {
    width: 100%;
    border-collapse: collapse;
}
.dataframe th, .dataframe td {
    padding: 8px;
    text-align: left;
    border: 1px solid #00e6e6;
    white-space: normal !important;
    max-width: 150px;
}
.dataframe th {
    background: rgba(0, 230, 230, 0.1);
}
#status-message {
    text-align: center;
    color: #00ffcc;
    margin: 10px 0;
}
.slider-container .slider {
    background: rgba(10, 10, 40, 0.8);
    border: 2px solid #00e6e6;
}
.plot-container {
    margin-top: 20px;
    background: rgba(10, 10, 40, 0.8);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    padding: 10px;
}
"""

# Create Gradio interface with real-time graph and progress bar
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Neural Identity Matrix")
    gr.Markdown("Generate futuristic clone identities with an evolving AI core.")

    num_identities = gr.Slider(minimum=1, maximum=250, value=10, step=1, label="Number of Identities to Generate")
    resume_training = gr.Checkbox(label="Resume Training from Checkpoint", value=False)

    with gr.Row():
        generate_button = gr.Button("Initialize Identity Generation")
        clear_button = gr.Button("Clear Output")

    progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progress", interactive=False)
    status_message = gr.Markdown("Ready to Generate")
    loss_plot = gr.Plot(label="Training Loss")
    output = gr.Dataframe(label="Identity Matrix Output")
    download_button = gr.File(label="Download Identities as CSV", visible=False)

    generate_button.click(
        fn=generate_identities_gui_wrapper,
        inputs=[num_identities, resume_training],
        outputs=[output, download_button, progress_bar, status_message, loss_plot],
        queue=True
    )

# Launch the Gradio app
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    debug=True,
    height=800
)
