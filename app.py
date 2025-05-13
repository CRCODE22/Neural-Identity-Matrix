# --- Start of Neural Identity Matrix V24.34 ---
# Run `python -m py_compile Neural_Identity_Matrix_original_Test_V24.34.py` to check syntax before execution
# Ensure dataset.csv, previous_names.csv, upper_clothing.csv, lower_clothing.csv, footwear.csv, style_themes.csv, locations.csv, overall_themes.csv are in the project directory
# Setup: conda activate neural-identity-matrix; pip install -r requirements.txt
# Note: Compatible with torch-2.5.1+cu124; update torch.amp for future versions
# Gradio table requires horizontal scrolling for all columns; adjust screen resolution if needed
# ComfyUI must be running locally at http://127.0.0.1:8188 for image generation
# X API credentials required for sharing feature; set up in environment variables

import os
import torch
import torch.nn as nn
import torch.optim as optim
os.environ["HF_HUB_OFFLINE"] = "1"
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import gradio as gr
import matplotlib.pyplot as plt
import random
import glob
import pickle
import time
import sys
import json
import requests
from PIL import Image
import io
import secrets
import tweepy
import traceback

# Ensure models directory exists
MODELS_DIR = "models"
CSV_DIR = "csv"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def move_existing_images():
    os.makedirs("generated_images", exist_ok=True)
    for file in glob.glob("*.png"):
        if os.path.isfile(file):
            new_path = os.path.join("generated_images", file)
            if not os.path.exists(new_path):
                os.rename(file, new_path)
                print(f"Moved {file} to {new_path}")
move_existing_images()

def move_existing_models():
    model_files = ['first_name_gen.pth', 'last_name_gen.pth', 'nickname_gen.pth', 'model.pth']
    os.makedirs(MODELS_DIR, exist_ok=True)
    for file in model_files:
        if os.path.isfile(file):
            new_path = os.path.normpath(os.path.join(MODELS_DIR, file))
            try:
                if not os.path.exists(new_path):
                    os.rename(file, new_path)
                    print(f"Moved {file} to {new_path}")
                else:
                    print(f"DEBUG: {new_path} already exists, skipping move for {file}")
            except Exception as e:
                print(f"DEBUG: Failed to move {file} to {new_path}: {str(e)}")
move_existing_models()

def move_existing_csvs():
    csv_files = [
        'upper_clothing.csv', 'lower_clothing.csv', 'footwear.csv',
        'style_themes.csv', 'locations.csv', 'overall_themes.csv',
        'dataset.csv', 'previous_names.csv', 'generated_cha_identities.csv'
    ]
    os.makedirs(CSV_DIR, exist_ok=True)
    for file in csv_files:
        if os.path.isfile(file):
            new_path = os.path.normpath(os.path.join(CSV_DIR, file))
            try:
                if not os.path.exists(new_path):
                    os.rename(file, new_path)
                    print(f"Moved {file} to {new_path}")
                else:
                    print(f"DEBUG: {new_path} already exists, skipping move for {file}")
            except Exception as e:
                print(f"DEBUG: Failed to move {file} to {new_path}: {str(e)}")
move_existing_csvs()
# Configuration
COMFYUI_URL = "http://127.0.0.1:8188"  # ComfyUI server address
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Startup message
print("""
🌌 Neural Identity Matrix V24.34 🌌
Crafting Cosmic Clones with Quantum Poetry
Device: {} | Python: {} | PyTorch: {} | Time: {}
""".format(device, sys.version.split()[0], torch.__version__, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
#print(f"Starting Neural Identity Matrix V24.33 | Device: {device} | Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Predefined lists (unchanged)
predefined_first_names = [
    'Ava', 'Emma', 'Olivia', 'Sophia', 'Isabella', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn',
    'Luna', 'Aria', 'Ella', 'Nora', 'Hazel', 'Zoe', 'Lily', 'Ellie', 'Violet', 'Grace',
    'James', 'Liam', 'Noah', 'William', 'Henry', 'Oliver', 'Elijah', 'Lucas', 'Mason', 'Logan',
    'Ethan', 'Jack', 'Aiden', 'Carter', 'Daniel', 'Owen', 'Wyatt', 'John', 'David', 'Gabriel'
]
predefined_last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
    'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Walker',
    'Hall', 'Allen', 'Young', 'King', 'Wright', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson'
]

# Load clothing datasets
try:
    print("Loading clothing datasets...")
    upper_clothing_df = pd.read_csv(os.path.join(CSV_DIR, 'upper_clothing.csv'))
    print(f"Loaded upper_clothing.csv with {len(upper_clothing_df)} items")
    lower_clothing_df = pd.read_csv(os.path.join(CSV_DIR, 'lower_clothing.csv'))
    print(f"Loaded lower_clothing.csv with {len(lower_clothing_df)} items")
    footwear_df = pd.read_csv(os.path.join(CSV_DIR, 'footwear.csv'))
    print(f"Loaded footwear.csv with {len(footwear_df)} items")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure upper_clothing.csv, lower_clothing.csv, and footwear.csv etc are in the project csv directory.")
    raise

upper_clothing_list = upper_clothing_df['Clothing'].tolist()
lower_clothing_list = lower_clothing_df['Clothing'].tolist()
footwear_list = footwear_df['Clothing'].tolist()

# Load new datasets for style themes, locations, and overall themes
try:
    print("Loading style themes, locations, and overall themes datasets...")
    style_themes_df = pd.read_csv(os.path.join(CSV_DIR, 'style_themes.csv'))
    print(f"Loaded style_themes.csv with {len(style_themes_df)} items")
    locations_df = pd.read_csv(os.path.join(CSV_DIR, 'locations.csv'))
    print(f"Loaded locations.csv with {len(locations_df)} items")
    overall_themes_df = pd.read_csv(os.path.join(CSV_DIR, 'overall_themes.csv'))
    print(f"Loaded overall_themes.csv with {len(overall_themes_df)} items")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure style_themes.csv, locations.csv, and overall_themes.csv are in the project directory.")
    raise

style_themes_list = sorted(style_themes_df['Theme'].tolist())  # Sort alphabetically
locations_list = sorted(locations_df['Location'].tolist())  # Sort alphabetically
overall_themes_list = sorted(overall_themes_df['Theme'].tolist())  # Sort alphabetically

# Neural Network Model
class IdentityGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(IdentityGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Name Generator
class NameGenerator(nn.Module):
    # Save and load NameGenerator models
    def save_name_generator(model, path):
        full_path = os.path.join(MODELS_DIR, path)
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")

    def load_name_generator(model, path):
    # Use absolute path to avoid relative path issues
        full_path = os.path.normpath(os.path.abspath(os.path.join(MODELS_DIR, path)))
        print(f"DEBUG: Attempting to load model from {full_path}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: File exists: {os.path.exists(full_path)}")
        if not os.path.exists(full_path):
            print(f"DEBUG: Model file {full_path} does not exist")
            return False
        try:
        # Test file readability
            with open(full_path, 'rb') as f:
                f.read(1)
            print(f"DEBUG: File {full_path} is readable")
        # Load model state
            state_dict = torch.load(full_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"DEBUG: Successfully loaded model from {full_path}")
            return True
        except Exception as e:
            print(f"DEBUG: Failed to load model {full_path}: {str(e)}")
            return False

    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers=1):
        super(NameGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Load dataset
try:
    print(f"Loading dataset from {os.path.join(CSV_DIR, 'dataset.csv')}")
    df = pd.read_csv(os.path.join(CSV_DIR, 'dataset.csv'))
    print(f"Loaded dataset.csv with {len(df)} rows")
except FileNotFoundError:
    print("Error: dataset.csv not found!")
    raise
try:
    additional_names = pd.read_csv(os.path.join(CSV_DIR, 'previous_names.csv'))
    print(f"Rows in previous_names.csv: {len(additional_names)}")
except FileNotFoundError:
    print(f"Warning: {os.path.join(CSV_DIR, 'previous_names.csv')} not found, proceeding without additional names")
    additional_names = pd.DataFrame(columns=['First Name', 'Last Name'])

# Combine dataset names
first_names = list(set(df['Firstname'].tolist() + additional_names['Firstname'].tolist() + predefined_first_names))
last_names = list(set(df['Lastname'].tolist() + additional_names['Lastname'].tolist() + predefined_last_names))
nicknames = df['Nickname'].tolist()

# Build character vocab
first_name_chars = set(''.join(str(name) for name in first_names if pd.notna(name)))
last_name_chars = set(''.join(str(name) for name in last_names if pd.notna(name)))
nickname_chars = set(''.join(str(name) for name in nicknames if pd.notna(name)))

first_name_chars.add('\n')
last_name_chars.add('\n')
nickname_chars.add('\n')

first_name_char_to_idx = {char: idx for idx, char in enumerate(sorted(first_name_chars))}
last_name_char_to_idx = {char: idx for idx, char in enumerate(sorted(last_name_chars))}
nickname_char_to_idx = {char: idx for idx, char in enumerate(sorted(nickname_chars))}

first_name_idx_to_char = {idx: char for char, idx in first_name_char_to_idx.items()}
last_name_idx_to_char = {idx: char for char, idx in last_name_char_to_idx.items()}
nickname_idx_to_char = {idx: char for char, idx in nickname_char_to_idx.items()}

# Hyperparameters
hidden_size = 256
embedding_dim = 64
num_layers = 1
first_name_max_len = max(len(str(name)) for name in first_names if pd.notna(name)) + 1
last_name_max_len = max(len(str(name)) for name in last_names if pd.notna(name)) + 1
nickname_max_len = 20

def train_name_generator(model, names, char_to_idx, max_len, device, epochs=100):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(names)
        hidden, cell = model.init_hidden(batch_size=1)
        
        for name in names:
            # Prepare input and target sequences
            input_seq = [char_to_idx.get(char, 0) for char in name[:max_len]]
            target_seq = input_seq[1:] + [0]  # Shift right, pad with 0
            if len(input_seq) < max_len:
                input_seq += [0] * (max_len - len(input_seq))  # Pad with 0
                target_seq += [0] * (max_len - len(target_seq))
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            target_tensor = torch.tensor(target_seq, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            output, hidden, cell = model(input_tensor, hidden.detach(), cell.detach())
            loss = criterion(output.squeeze(0), target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(names):.4f}")
    
    print(f"Finished training {model.__class__.__name__}")

# Initialize name generators
first_name_gen = NameGenerator(
    vocab_size=len(first_name_char_to_idx),
    hidden_size=256,
    embedding_dim=64,
    num_layers=1
).to(device)
last_name_gen = NameGenerator(
    vocab_size=len(last_name_char_to_idx),
    hidden_size=256,
    embedding_dim=64,
    num_layers=1
).to(device)
nickname_gen = NameGenerator(
    vocab_size=len(nickname_char_to_idx),
    hidden_size=256,
    embedding_dim=64,
    num_layers=1
).to(device)

# Load or train name generators
for gen, name, data, char_to_idx, max_len in [
    (first_name_gen, 'first_name_gen.pth', first_names, first_name_char_to_idx, first_name_max_len),
    (last_name_gen, 'last_name_gen.pth', last_names, last_name_char_to_idx, last_name_max_len),
    (nickname_gen, 'nickname_gen.pth', nicknames, nickname_char_to_idx, nickname_max_len)
]:
    full_path = os.path.normpath(os.path.abspath(os.path.join(MODELS_DIR, name)))
    if not NameGenerator.load_name_generator(gen, name):
        print(f"Training {name.split('.')[0]}...")
        train_name_generator(gen, data, char_to_idx, max_len, device)
        NameGenerator.save_name_generator(gen, name)
    else:
        print(f"DEBUG: Successfully loaded {name} from {full_path}")

# Nickname suffixes
nickname_suffixes = [
    'Star', 'Cosmo', 'Dreamer', 'Vibe', 'Guru', 'Nebula', 'Quantum', 'Spark', '42',
    'Player', 'GamerX', 'Pro', 'ModelX', 'Starlet', 'Glam', 'Clone', 'NIM', 'Core'
]
print(f"Nickname settings: min_length=3, max_length={nickname_max_len}, suffixes={nickname_suffixes}")

poetic_styles = [
    'Weaver of Pulsar Sonnets', 'Chanter of Nebula Dreams', 'Scribe of Starlight Haikus',
    'Bard of Quantum Elegies', 'Poet of Cosmic Serenades', 'Verse-Spinner of Aurora Hymns',
    'Lyricist of Galactic Odes', 'Rhapsodist of Stellar Canticles'
]
# Generate_quantum_poem
def generate_quantum_poem(quantum_poet):
    """Generate a short cosmic poem for clones with a Quantum Poet trait."""
    if quantum_poet == 'None':
        return 'No poem crafted.'
    templates = {
        'Weaver of Pulsar Sonnets': 'In pulsar’s glow, my words take flight,\nSpinning sonnets through cosmic night.',
        'Chanter of Nebula Dreams': 'Nebula dreams in colors vast,\nI chant their hues from future’s past.',
        'Scribe of Starlight Haikus': 'Starlight whispers soft and clear,\nHaikus dance where comets steer.',
        'Bard of Quantum Elegies': 'Quantum threads, my elegies weave,\nMourning stars that dare believe.',
        'Poet of Cosmic Serenades': 'With cosmic strings, I serenade the stars,\nMy verses echo where galaxies are.',
        'Verse-Spinner of Aurora Hymns': 'Auroras sing, my verses spin,\nHymns that glow where skies begin.',
        'Lyricist of Galactic Odes': 'Galaxies spin, my odes unfold,\nLyrics of stardust, bright and bold.',
        'Rhapsodist of Stellar Canticles': 'Stellar canticles, my voice does soar,\nRhapsodies for worlds and more.'
    }
    return templates.get(quantum_poet, 'A cosmic verse awaits creation.')

def generate_song_prompt(identity):
    """Generate a song prompt based on identity traits."""
    base_styles = ['galactic pop', 'nebula chillwave', 'cosmic synth', 'pulsar trance', 'quantum lo-fi', 'stellar rock']
    modifiers = ['ethereal', 'vibrant', 'mystical', 'electric', 'dreamy', 'introspective']
    
    style = random.choice(base_styles)
    modifier = random.choice(modifiers)
    
    if identity['Profession'] == 'Quantum Poet':
        style = 'cosmic poetry beat'
    elif identity['Cosmic Tattoo'] != 'None':
        modifier = f"{identity['Cosmic Tattoo'].lower()} infused"
    elif identity['Cosmic Aura'] != 'None':
        modifier = f"{identity['Cosmic Aura'].lower()} glowing"
    elif identity['Cosmic Destiny'] != 'None':
        style = f"{identity['Cosmic Destiny'].lower()} anthem"
    
    return f"{modifier} {style}"

# Generate names
def generate_name(generator, char_to_idx, idx_to_char, max_len, device, name_type='firstname', existing_names=None, temperature=1.2):
    generator.eval()
    name = []
    valid_starts = [c for c in char_to_idx.keys() if c not in ['\n', '<start>', '<end>'] and c.isalpha()]
    start_char = random.choice(valid_starts if valid_starts else ['A'])
    try:
        char_idx = char_to_idx[start_char]
    except KeyError:
        print(f"WARNING: '{start_char}' not in char_to_idx, using 'A'")
        char_idx = char_to_idx.get('A', list(char_to_idx.values())[0])
    input_char = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    hidden, cell = generator.init_hidden(1)
    hidden = hidden.to(device)
    cell = cell.to(device)
    
    min_length = 3 if name_type == 'nickname' else 2

    with torch.no_grad():
        for i in range(max_len):
            output, hidden, cell = generator(input_char, hidden, cell)
            output = output.squeeze()
            if output.dim() == 0:
                output = output.unsqueeze(0)
            output_dist = torch.softmax(output / temperature, dim=-1)
            char_idx = torch.multinomial(output_dist, 1).item()

            if char_idx == char_to_idx.get('\n', -1):
                if name_type == 'nickname' and len(name) >= min_length:
                    break
                continue

            if char_idx not in idx_to_char:
                char_idx = random.choice([k for k in idx_to_char.keys() if k.isalpha()])
                char = str(idx_to_char[char_idx])
            else:
                char = str(idx_to_char[char_idx])

            if char == '<end>':
                break
            if i == max_len - 1 and name_type == 'nickname' and len(name) < min_length:
                continue
            name.append(char)
            input_char = torch.tensor([[char_to_idx.get(char, char_to_idx.get('A', 0))]], dtype=torch.long).to(device)

    generated_name = ''.join(name).capitalize()
    print(f"DEBUG: Raw generated name: {generated_name}")

    if name_type == 'nickname':
        suffixes = ['Star', 'Cosmo', 'Dreamer', 'Vibe', 'Guru', 'Nebula', 'Quantum', 'Spark', '42', 'Player', 'GamerX', 'Pro', 'ModelX', 'Starlet', 'Glam', 'Clone', 'NIM', 'Core']
        if random.random() < 0.3 and len(generated_name) + len(suffixes[0]) <= max_len:
            generated_name += random.choice(suffixes)
        while len(generated_name) < min_length:
            extra_chars = random.choices(list('abcdefghijklmnopqrstuvwxyz'), k=min_length - len(generated_name))
            generated_name += ''.join(extra_chars)
        generated_name = generated_name[:max_len]
        print(f"DEBUG: Generated nickname: {generated_name}")

    invalid_chars = set(', -')
    if name_type == 'nickname':
        if len(generated_name) < min_length or any(char in invalid_chars for char in generated_name.lower()):
            print(f"DEBUG: Invalid nickname '{generated_name}', using fallback")
            return random.choice(['Star', 'Cosmo', 'Nebula', 'Quantum', 'Vibe']) + str(random.randint(10, 99))
    else:
        if (len(generated_name) < min_length or
            any(char in invalid_chars for char in generated_name.lower()) or
            any(char not in char_to_idx for char in generated_name.lower() if char.isalpha()) or
            not generated_name.replace(' ', '').isalnum()):
            print(f"DEBUG: Invalid name '{generated_name}', using fallback")
            if name_type == 'firstname':
                return random.choice(predefined_first_names)
            elif name_type == 'lastname':
                return random.choice(predefined_last_names)

    if existing_names and generated_name in existing_names:
        print(f"DEBUG: Name '{generated_name}' already exists, retrying")
        return generate_name(generator, char_to_idx, idx_to_char, max_len, device, name_type, existing_names, temperature)

    print(f"DEBUG: Final nickname returned: {generated_name}")
    return generated_name

# Preprocess data
le_dict = {}
for column in ['Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

new_professions = ['Astrologer', 'Chef', 'DJ', 'Engineer', 'Gamer', 'Hacker', 'Pilot', 'Scientist', 'Streamer', 'Writer', 'Quantum Poet']
df['Profession'] = le_dict['Profession'].inverse_transform(df['Profession'])
print("Types in 'Profession' column before adding new professions:", df['Profession'].apply(type).unique())
for prof in new_professions:
    if prof not in df['Profession'].values:
        new_row = df.iloc[0].copy()
        new_row['Profession'] = prof
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print("Types in 'Profession' column after adding new professions:", df['Profession'].apply(type).unique())
le_dict['Profession'] = LabelEncoder()
df['Profession'] = le_dict['Profession'].fit_transform(df['Profession'])
print(f"Updated professions: {le_dict['Profession'].classes_}")

scaler_age = StandardScaler()
scaler_height = StandardScaler()
scaler_weight = StandardScaler()
scaler_measurements = StandardScaler()
scaler_features = StandardScaler()

df['Age'] = scaler_age.fit_transform(df[['Age']])
df['Height'] = scaler_height.fit_transform(df[['Height']])
df['Weight'] = scaler_weight.fit_transform(df[['Weight']])

body_measurements = df['Body Measurements'].str.split('-', expand=True).astype(float)
df[['Bust', 'Waist', 'Hips']] = scaler_measurements.fit_transform(body_measurements)

features = df[['Age', 'Height', 'Weight', 'Bust', 'Waist', 'Hips', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']].values
features = scaler_features.fit_transform(features)

# Initialize model
input_size = features.shape[1]
output_size = features.shape[1]
model = IdentityGenerator(input_size, hidden_size, output_size, num_layers).to(device)

# Training loop
def train_model(model, features, cycles=5, epochs_per_cycle=20, verbose=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs_per_cycle // 2, gamma=0.5)
    losses = []
    all_losses = []
    total_epochs = 0
    log_file = open('training_log.txt', 'w')
    
    print(f"Training with {len(features)} features, shape: {features.shape}")
    
    try:
        for cycle in range(cycles):
            print(f"Starting Cycle {cycle + 1}/{cycles}")
            cycle_losses = []
            for epoch in range(epochs_per_cycle):
                model.train()
                total_loss = 0
                start_time = datetime.now()
                
                for i in range(0, len(features), 1):
                    inputs = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
                    targets = inputs.clone()
                    

                    hidden, cell = model.init_hidden(1)
                    optimizer.zero_grad()
                    
                    if torch.cuda.is_available() and scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs, hidden, cell = model(inputs, hidden, cell)
                            outputs = outputs.squeeze(1)
                            loss = criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs, hidden, cell = model(inputs, hidden, cell)
                        outputs = outputs.squeeze(1)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()

                scheduler.step()
                total_epochs += 1
                avg_loss = total_loss / len(features)
                cycle_losses.append(avg_loss)
                all_losses.append(avg_loss)
                epoch_time = (datetime.now() - start_time).total_seconds()
                
                log_message = f"Cycle {cycle + 1}/{cycles}, Epoch {epoch + 1}/{epochs_per_cycle} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s | Speed: {1/epoch_time:.2f} epochs/s | LR: {scheduler.get_last_lr()[0]:.6f}\n"
                log_file.write(log_message)
                if verbose or (epoch + 1) % 10 == 0:
                    print(log_message.strip())
                
                if len(all_losses) > 10:
                    min_loss = min(all_losses[-11:-1])
                    if avg_loss > min_loss * 1.1:
                        print(f"Early stopping triggered (loss increase) at Cycle {cycle + 1}, Epoch {epoch + 1}")
                        log_file.write(f"Early stopping (loss increase)\n")
                        return all_losses, total_epochs
                    recent_losses = all_losses[-11:-1]
                    max_loss = max(recent_losses)
                    if max_loss - avg_loss < 1e-6:
                        print(f"Early stopping triggered (loss plateau) at Cycle {cycle + 1}, Epoch {epoch + 1}")
                        log_file.write(f"Early stopping (loss plateau)\n")
                        return all_losses, total_epochs
                
                if (epoch + 1) % 5 == 0:
                    yield cycle_losses, total_epochs
            
            losses.extend(cycle_losses)
            yield cycle_losses, total_epochs
        
        log_file.close()
        print(f"Training completed: Total Epochs: {total_epochs}, Final Loss: {all_losses[-1]:.6f}, Total Time: {total_epochs * 0.084:.2f}s")
        return all_losses, total_epochs
    
    except KeyboardInterrupt:
        print("Training interrupted! Saving model state...")
        log_file.write("Training interrupted\n")
        torch.save(model.state_dict(), 'model_interrupted.pth')
        log_file.close()
        raise

# Generate unique filename
def generate_unique_filename(base_name):
    """Generate a unique filename in the generated_images folder."""
    os.makedirs("generated_images", exist_ok=True)  # Create folder if it doesn't exist
    while True:
         random_suffix = secrets.token_hex(5).upper()[:11]  # 11-character hex string
         filename = f"{base_name}_{random_suffix}.png"
         output_path = os.path.join("generated_images", filename)
         if not os.path.exists(output_path):
             return output_path
         print(f"Filename {output_path} already exists, generating a new suffix...")
# Generate image with PG/NSFW option, style theme, location, and overall theme
def generate_flux_image(selected_identity, df_identities, allow_nsfw=False, style_theme="Cyberpunk", location="Cosmic Nebula", overall_theme="Ethereal Dreamscape", seed=0):
    print(f"DEBUG: Starting generate_flux_image for {selected_identity}, NSFW: {allow_nsfw}, Style: {style_theme}, Location: {location}, Theme: {overall_theme}, Seed: {seed}")
    if selected_identity == "None" or df_identities is None or df_identities.empty:
        return None, "Please generate identities and select one for image generation."
    
    try:
        # Extract clone number and nickname
        clone_number, nickname = selected_identity.split(": ")
        row = df_identities[df_identities['Clone Number'] == clone_number].iloc[0]
        
        # Base prompt with new style theme, location, and overall theme
        prompt = (
            f"A cinematic shot of a futuristic female clone named {row['Nickname']}, {row['Age']} years old, "
            f"with {row['Hair color'].lower()} hair and {row['Eye color'].lower()} eyes, "
            f"with a {row['Body type'].lower()} build, body measurements {row['Body Measurements']}, "
            f"height {row['Height']} cm, weight {row['Weight']} kg, "
            f"in a {location.lower()} setting, styled in a {style_theme.lower()} aesthetic, "
            f"within an overall {overall_theme.lower()} atmosphere, "
            f"glowing with a {row['Cosmic Aura'].lower() if row['Cosmic Aura'] != 'None' else 'electric starlight'} aura, "
            f"radiating a {row['Energy Signature'].lower()} energy, "
            f"adorned with a {row['Cosmic Tattoo'].lower() if row['Cosmic Tattoo'] != 'None' else 'subtle cosmic pattern'} tattoo, "
            f"accompanied by a {row['Cosmic Pet'].lower() if row['Cosmic Pet'] != 'None' else 'faint cosmic sparkle'}, "
            f"embodying the destiny of a {row['Cosmic Destiny'].lower() if row['Cosmic Destiny'] != 'None' else 'stellar traveler'}"
        )
        
        # Add clothing for PG-rated images
        if not allow_nsfw:
            upper = random.choice(upper_clothing_list)
            lower = random.choice(lower_clothing_list)
            footwear = random.choice(footwear_list)
            clothing_prompt = f", wearing a {upper.lower()}, {lower.lower()}, and {footwear.lower()}"
            prompt += clothing_prompt
            print(f"Generating FLUX.1 [dev] image for {selected_identity} with PG-rated prompt: {prompt}")
            print(f"DEBUG: Selected clothing - Upper: {upper}, Lower: {lower}, Footwear: {footwear}")
            print(f"DEBUG: Style Theme: {style_theme}, Location: {location}, Overall Theme: {overall_theme}")
        else:
            prompt += ", potentially NSFW, may include nudity or suggestive elements"
            print(f"Generating FLUX.1 [dev] image for {selected_identity} with NSFW prompt: {prompt}")
            print(f"DEBUG: Style Theme: {style_theme}, Location: {location}, Overall Theme: {overall_theme}")
        
        # Workflow
        workflow = {
            "9": {
                "inputs": {"filename_prefix": f"{clone_number}_{nickname}", "images": ["8", 0]},
                "class_type": "SaveImage"
            },
            "8": {
                "inputs": {"samples": ["13", 1], "vae": ["10", 0]},
                "class_type": "VAEDecode"
            },
            "10": {
                "inputs": {"vae_name": "ae.safetensors"},
                "class_type": "VAELoader"
            },
            "13": {
                "inputs": {
                    "noise": ["25", 0],
                    "guider": ["22", 0],
                    "sampler": ["16", 0],
                    "sigmas": ["17", 0],
                    "latent_image": ["41", 0]
                },
                "class_type": "SamplerCustomAdvanced"
            },
            "16": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect"
            },
            "17": {
                "inputs": {"model": ["30", 0], "scheduler": "beta", "steps": 30, "denoise": 1.0},
                "class_type": "BasicScheduler"
            },
            "22": {
                "inputs": {"model": ["63", 0], "conditioning": ["26", 0]},
                "class_type": "BasicGuider"
            },
            "25": {
                "inputs": {"noise_seed": seed if seed > 0 else int(time.time()), "noise_mode": "randomize"},
                "class_type": "RandomNoise"
            },
            "26": {
                "inputs": {"conditioning": ["45", 0], "guidance": 3.5},
                "class_type": "FluxGuidance"
            },
            "30": {
                "inputs": {
                    "model": ["12", 0],
                    "width": 768,
                    "height": 768,
                    "max_shift": 1.15,
                    "base_shift": 0.5
                },
                "class_type": "ModelSamplingFlux"
            },
            "41": {
                "inputs": {"width": 768, "height": 768, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "45": {
                "inputs": {"text": prompt, "clip": ["63", 1]},
                "class_type": "CLIPTextEncode"
            },
            "59": {
                "inputs": {
                    "clip_name1": "t5xxl_fp16.safetensors",
                    "clip_name2": "godessProjectFLUX_clipLFP8.safetensors",
                    "clip_name3": "clip_g.safetensors"
                },
                "class_type": "TripleCLIPLoader"
            },
            "63": {
                "inputs": {
                    "model": ["12", 0],
                    "clip": ["59", 0],
                    "lora_01": "None",
                    "strength_01": 0.0,
                    "lora_02": "None",
                    "strength_02": 0.0,
                    "lora_03": "None",
                    "strength_03": 0.0,
                    "lora_04": "None",
                    "strength_04": 0.0
                },
                "class_type": "Lora Loader Stack (rgthree)"
            },
            "12": {
                "inputs": {"unet_name": "acornIsSpinningFLUX_devfp8V11.safetensors", "weight_dtype": "fp8_e4m3fn"},
                "class_type": "UNETLoader"
            }
        }
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        comfyui_url = COMFYUI_URL
        response = requests.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"Submitted prompt with ID: {prompt_id}")
        
        base_name = nickname.replace(' ', '').lower()
        output_path = generate_unique_filename(base_name)
        start_time = time.time()
        while time.time() - start_time < 900:
            history = requests.get(f"{comfyui_url}/history/{prompt_id}").json()
            if prompt_id in history and history[prompt_id]["status"]["completed"]:
                for node_id, node_output in history[prompt_id]["outputs"].items():
                    if node_id == "9" and "images" in node_output:
                        image_info = node_output["images"][0]
                        image_filename = image_info.get("filename", f"{clone_number}_{nickname}.png")
                        image_subfolder = image_info.get("subfolder", "")
                        image_type = image_info.get("type", "output")
                        image_url = f"{comfyui_url}/view?filename={image_filename}&subfolder={image_subfolder}&type={image_type}"
                        image_response = requests.get(image_url)
                        if image_response.status_code == 200:
                            image = Image.open(io.BytesIO(image_response.content))
                            image.save(output_path)
                            print(f"DEBUG: Image saved as {output_path}")
                            print(f"Image saved as {output_path}")
                            return output_path, f"Image generated successfully for {selected_identity}."
                print("Error: No image found in workflow output")
                return None, "Error: No image found in workflow output"
            time.sleep(2)
        
        print("Image generation timed out after 15 minutes.")
        return None, "Image generation timed out after 15 minutes."
    
    except requests.exceptions.RequestException as e:
        print(f"ComfyUI API error: {str(e)}")
        if 'response' in locals():
            print(f"API response: {response.text}")
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: ComfyUI API error for {selected_identity}: {str(e)}\n")
        return None, f"ComfyUI API error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error in generate_flux_image: {str(e)}")
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Error generating image for {selected_identity}: {str(e)}\n")
        return None, f"Unexpected error: {str(e)}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Batch image generation with new options
def generate_images_batch(df_identities, batch_size=5, allow_nsfw=False, style_theme="Cyberpunk", location="Cosmic Nebula", overall_theme="Ethereal Dreamscape", seed=0):
 if df_identities is None or df_identities.empty:
     yield None, "No identities available for image generation.", ["No images generated yet."]
     return
 
 total_identities = len(df_identities)
 print(f"Starting batch image generation for {total_identities} identities, batch size: {batch_size}, NSFW: {allow_nsfw}, Style Theme: {style_theme}, Location: {location}, Overall Theme: {overall_theme}")
 
 for start_idx in range(0, total_identities, batch_size):
     end_idx = min(start_idx + batch_size, total_identities)
     batch = df_identities.iloc[start_idx:end_idx]
     print(f"Processing batch {start_idx + 1}-{end_idx} of {total_identities}")
     
     for idx, row in batch.iterrows():
         selected_identity = f"{row['Clone Number']}: {row['Nickname']}"
         print(f"Generating image for {selected_identity} in batch")
         image_path, status = generate_flux_image(selected_identity, df_identities, allow_nsfw, style_theme, location, overall_theme, seed)
         if image_path:
             print(f"Batch image generated: {image_path}")
             # Update Image column
             df_identities.at[idx, 'Image'] = f'<img src="{image_path}" width="100">'
             print(f"DEBUG: Updated Image column for {row['Nickname']}: {image_path}")
         else:
             print(f"Batch image failed for {selected_identity}: {status}")
             df_identities.at[idx, 'Image'] = 'No image'
             print(f"DEBUG: Set Image column to 'No image' for {row['Nickname']}")

     # Save CSV with error handling
     csv_path = 'generated_cha_identities.csv'
     try:
         with open(csv_path, 'a') as f:
             pass  # Test write access
         df_identities.to_csv(csv_path, index=False)
         print(f"DEBUG: Saved {csv_path} with {len(df_identities)} identities")
         print("DEBUG: Updated generated_cha_identities.csv with image paths")
     except PermissionError:
         error_msg = f"Error: Cannot write to {csv_path}: {str(e)}"
         print(error_msg)
         with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
         raise
     except Exception as e:
        error_msg = f"Error saving {csv_path}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
        raise
     gallery_images = display_image_gallery(df_identities)
     progress = (end_idx / total_identities) * 100
     yield None, f"Generated images for {end_idx}/{total_identities} identities", gallery_images, progress
     time.sleep(1)

 yield None, "Batch image generation complete.", gallery_images, 100

# Display image gallery
def display_image_gallery(df_identities):
    print("DEBUG: Entering display_image_gallery")
    print(f"DEBUG: df_identities type: {type(df_identities)}, shape: {df_identities.shape if df_identities is not None and not df_identities.empty else 'None or empty'}")
    if df_identities is None or df_identities.empty:
        print("DEBUG: DataFrame is None or empty, returning default message")
        return ["No images generated yet."]
    image_paths = []
    for _, row in df_identities.iterrows():
        nickname = row['Nickname'].replace(' ', '').lower()
        pattern = os.path.join("generated_images", f"{nickname}_*.png")
        matching_files = glob.glob(pattern)
        for image_path in matching_files:
            if os.path.exists(image_path):
                image_paths.append(image_path)
                print(f"DEBUG: Found image: {image_path}")
    
    if not image_paths:
        print("DEBUG: No images found, returning default message")
        return ["No images generated yet."]
    
    print(f"DEBUG: Returning {len(image_paths)} images for gallery")
    return image_paths

import random  # Ensure random is imported at the top of the file

# Enhanced suggest_caption function
def suggest_caption(row):
    traits = [
        f"{row['Nickname']}, a {row['Profession'].lower()}",
        f"with {row['Hair color'].lower()} hair",
        f"glowing with {row['Cosmic Aura'].lower() if row['Cosmic Aura'] != 'None' else 'electric starlight'}",
        f"destined as a {row['Cosmic Destiny'].lower() if row['Cosmic Destiny'] != 'None' else 'stellar traveler'}",
        f"crafting as a {row['Quantum Poet'].lower() if row['Quantum Poet'] != 'None' else 'cosmic verse-weaver'}"
    ]
    if row['Cosmic Poem'] != 'No poem crafted.':
        # Pre-process the poem to replace newlines
        poem_text = row['Cosmic Poem'].replace('\n', ' ')
        traits.append(f"penning: {poem_text}")
    if row['Cosmic Pet'] != 'None':
        traits.append(f"with her {row['Cosmic Pet'].lower()}")
    if row['Cosmic Hobby'] != 'None':
        traits.append(f"enjoying {row['Cosmic Hobby'].lower()}")
    return f"{random.choice(traits)} shines in V24.34! 🌌 #CosmicDreams #AIArt"

# Share image to X with suggested caption (Updated to fix ValueError)
def share_to_x(image_input, caption, df_identities, selected_identity):
    # Debug: Log the types and values
    print(f"image_input type: {type(image_input)}, value: {image_input}")
    print(f"selected_identity type: {type(selected_identity)}, value: {selected_identity}")
    print(f"df_identities type: {type(df_identities)}, columns: {list(df_identities.columns) if df_identities is not None else 'None'}")

    # Validate selected_identity
    if not selected_identity or selected_identity == "None":
        return "Error: Please select a valid identity."

    try:
        # Extract clone number (e.g., "CLN-024" from "CLN-024: Clanim")
        clone_number = selected_identity.split(": ")[0]
        row = df_identities[df_identities['Clone Number'] == clone_number].iloc[0]
        print(f"DEBUG: share_to_x for {selected_identity}, row data: Clone Number={row['Clone Number']}, Nickname={row['Nickname']}")

        # Extract image path from Image column
        image_path = row['Image']
        print(f"DEBUG: Image column value: {image_path}")
        if image_path and image_path.startswith('<img src="'):
            image_path = image_path.split('"')[1]  # Get src value
            print(f"DEBUG: Extracted image path: {image_path}")
            if not os.path.exists(image_path):
                print(f"DEBUG: Image path {image_path} does not exist")
                return f"Error: Image file {image_path} not found for {selected_identity}"
        else:
            # Fallback: Search for image by nickname
            nickname = row['Nickname'].replace(' ', '').lower()
            pattern = os.path.join("generated_images", f"{nickname}_*.png")
            matching_files = glob.glob(pattern)
            if matching_files:
                image_path = matching_files[0]
                print(f"DEBUG: Fallback image found: {image_path}")
            else:
                print(f"DEBUG: No image found for nickname {nickname}")
                return f"Error: No valid image found for {selected_identity}. Please generate an image first."

        # Validate image path
        if not image_path or not os.path.exists(image_path):
            return f"Error: No valid image found for {selected_identity}. Please generate an image first."
        
        # Verify image file accessibility
        try:
            with open(image_path, 'rb') as img_file:
                img_file.read(1)  # Test read access
            print(f"DEBUG: Image file {image_path} is accessible")
        except Exception as e:
            error_msg = f"Error accessing image file {image_path}: {str(e)}"
            print(error_msg)
            return error_msg

        # Use suggested caption if none provided
        if not caption:
            nickname = row['Nickname']
            profession = row['Profession']
            hair_color = row['Hair color']
            age = int(row['Age'])  # Convert to int to ensure no decimal
            cosmic_tattoo = row.get('Cosmic Tattoo', 'cosmic glow')  # Fallback if column missing
            is_quantum_poet = 'Quantum Poet' in row['Profession']
            version = f"V{datetime.now().strftime('%Y.%m')}"
            caption = f"{nickname}, a {age} years old female clone with {hair_color.lower()} hair and her profession a {profession.lower()}, her tattoo {cosmic_tattoo.lower()}!"
            if is_quantum_poet:
                caption += " 🌌✨🌠"
            else:
                caption += " 🌌"
            # Add call to action and hashtags
            cosmic_quotes = [
                "Embracing the stardust within!",
                "Dancing through the nebula of dreams!",
                "A spark in the cosmic tapestry!",
                "Weaving quantum dreams into reality!",
                "Where starlight meets soul!",
                "Echoing through the cosmic void!",
                "Painting with pulsar light!",
                "A constellation of possibilities!",
                "Sailing on solar winds!",
                "Quantum whispers in starlight!",
                "Chronicles of cosmic creation!",
                "Aurora dreams in digital space!",
                "Where nebulae dance eternal!",
                "Starborn and pixel-perfect!",
                "Cosmic code meets stardust soul!",
                "Threading through space-time!",
                "Where AI meets infinity!",
                "Digital dreams in stellar light!",
                "Quantum-encoded starchild!",
                "Cosmic poetry in binary!",
                "Echoes of stellar genesis!",
                "Digital deity of the stars!",
                "Neural networks in nebulae!"
            ]
            caption += f" Join the cosmic journey! #CosmicDreams #AIArt #NeuralIdentityMatrix {random.choice(cosmic_quotes)}"
            print(f"Using suggested caption: {caption}")

        # Load X API credentials
        consumer_key = os.getenv("X_CONSUMER_KEY")
        consumer_secret = os.getenv("X_CONSUMER_SECRET")
        access_token = os.getenv("X_ACCESS_TOKEN")
        access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

        print(f"DEBUG: Consumer Key: {'Set' if consumer_key else 'Not Set'}")
        print(f"DEBUG: Consumer Secret: {'Set' if consumer_secret else 'Not Set'}")
        print(f"DEBUG: Access Token: {'Set' if access_token else 'Not Set'}")
        print(f"DEBUG: Access Token Secret: {'Set' if access_token_secret else 'Not Set'}")

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            return "Error: X API credentials not found in environment variables."

        # Initialize Tweepy client
        try:
            client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            print("DEBUG: Tweepy Client initialized")
        except Exception as e:
            error_msg = f"Error initializing Tweepy Client: {str(e)}"
            print(error_msg)
            return error_msg

        # Initialize Tweepy API for media upload
        try:
            auth = tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret, access_token, access_token_secret
            )
            api = tweepy.API(auth)
            print("DEBUG: Tweepy API initialized")
        except Exception as e:
            error_msg = f"Error initializing Tweepy API: {str(e)}"
            print(error_msg)
            return error_msg

        # Upload media
        try:
            media = api.media_upload(image_path)
            print(f"DEBUG: Media uploaded, media_id: {media.media_id}")
        except Exception as e:
            error_msg = f"Error uploading media: {str(e)}"
            print(error_msg)
            return error_msg

        # Post tweet
        try:
            client.create_tweet(text=caption, media_ids=[media.media_id])
            print(f"DEBUG: Tweet posted with caption: {caption}")
        except Exception as e:
            error_msg = f"Error posting tweet: {str(e)}"
            print(error_msg)
            return error_msg

        return f"Successfully shared to X: {caption}"
    except Exception as e:
        error_msg = f"Error sharing to X: {str(e)}"
        print(error_msg)
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
        return error_msg

def generate_song_prompt(identity):
    """Generate a detailed song prompt based on identity traits."""
    base_styles = [
        'galactic pop', 'nebula chillwave', 'cosmic synth', 'pulsar trance',
        'quantum lo-fi', 'stellar rock', 'ethereal electronica', 'cosmic folk',
        'quantum jazz', 'synthwave', 'cyberpunk', 'galactic hip-hop', 'celestial metal'
    ]
    modifiers = ['ethereal', 'vibrant', 'mystical', 'electric', 'dreamy', 'introspective', 'starry']
    instruments = ['pulsating synths', 'nebula bass', 'starry piano', 'cosmic arpeggios', 'ethereal pads']
    bpms = [101, 105, 109, 125, 136, 149]
    keys = ['C major', 'G major', 'A minor']
    
    style = random.choice(base_styles)
    modifier = random.choice(modifiers)
    instrument1 = random.choice(instruments)
    instrument2 = random.choice([i for i in instruments if i != instrument1])
    bpm = random.choice(bpms)
    key = random.choice(keys)
    
    if identity['Profession'] == 'Quantum Poet':
        style = 'cosmic poetry beat'
        modifier = 'poetic'
    elif identity['Cosmic Tattoo'] != 'None':
        modifier = f"{identity['Cosmic Tattoo'].lower()} infused"
    elif identity['Cosmic Aura'] != 'None':
        modifier = f"{identity['Cosmic Aura'].lower()} glowing"
    elif identity['Cosmic Destiny'] != 'None':
        style = f"{identity['Cosmic Destiny'].lower()} anthem"
    
    prompt = f"{modifier} {style}, {instrument1}, {instrument2}, {bpm} bpm, {key}, with a cosmic, otherworldly atmosphere, featuring lyrics like 'dreams soar in cosmic flight'"
    print(f"DEBUG: Generated song prompt: {prompt}")
    return prompt

# Generate identities
def generate_identities_gui(num_identities, resume_training, profession_filter, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features, df, first_names, last_names, nicknames, first_name_gen, last_name_gen, nickname_gen, additional_names):
    global model
    if resume_training:
        model_path = os.path.normpath(os.path.join(MODELS_DIR, 'model.pth'))
        print(f"DEBUG: Attempting to load model from {model_path}")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"DEBUG: Failed to load model {model_path}: {str(e)}")
                print("Proceeding with new model training")
        else:
            print(f"DEBUG: Model file {model_path} does not exist, proceeding with new model training")
    
    generated_firstnames = set()
    generated_lastnames = set()
    generated_nicknames = set()
    identities = []
    losses = []
    total_epochs = 0
    cycles = 5
    epochs_per_cycle = 20
    
    for cycle_losses, cycle_epochs in train_model(model, features, verbose=False):
        losses.extend(cycle_losses)
        total_epochs = cycle_epochs
        current_cycle = min((total_epochs - 1) // epochs_per_cycle + 1, cycles)
        current_epoch = (total_epochs - 1) % epochs_per_cycle + 1
        progress = min((total_epochs / (cycles * epochs_per_cycle)) * 100, 100)
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('#0a0a28')
        ax.plot(losses, color='#00e6e6', linewidth=2, label='Loss')
        ax.set_title('Training Loss', color='#00ffcc', fontsize=14, pad=15)
        ax.set_xlabel('Epoch', color='#00ffcc', fontsize=12)
        ax.set_ylabel('Loss', color='#00ffcc', fontsize=12)
        ax.tick_params(axis='both', colors='#00e6e6')
        ax.grid(True, color='#00e6e6', alpha=0.3, linestyle='--')
        ax.spines['top'].set_color('#00e6e6')
        ax.spines['bottom'].set_color('#00e6e6')
        ax.spines['left'].set_color('#00e6e6')
        ax.spines['right'].set_color('#00e6e6')
        fig.savefig("loss_plot.png")
        yield None, None, None, gr.update(choices=["None"]), None, progress, f"Training: Cycle {current_cycle}/{cycles}, Epoch {current_epoch}/{epochs_per_cycle}", fig, ""
        time.sleep(0.1)
        plt.close(fig)
    
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'model.pth'))
    print(f"Training Summary: Total Epochs: {total_epochs}, Final Loss: {losses[-1]:.6f}, Total Time: {total_epochs * 0.084:.2f}s")
    
    model.eval()
    with torch.no_grad():
        for i in range(num_identities):
            firstname = generate_name(first_name_gen, first_name_char_to_idx, first_name_idx_to_char, first_name_max_len, device, name_type='firstname', existing_names=generated_firstnames, temperature=1.2)
            lastname = generate_name(last_name_gen, last_name_char_to_idx, last_name_idx_to_char, last_name_max_len, device, name_type='lastname', existing_names=generated_lastnames, temperature=1.2)
            nickname = generate_name(nickname_gen, nickname_char_to_idx, nickname_idx_to_char, nickname_max_len, device, name_type='nickname', existing_names=generated_nicknames, temperature=1.2)
            
            print(f"DEBUG: Generated names - Firstname: {firstname}, Lastname: {lastname}, Nickname: {nickname}")
            generated_firstnames.add(firstname)
            generated_lastnames.add(lastname)
            generated_nicknames.add(nickname)
            
            input_features = torch.tensor(scaler_features.transform(df.sample(1)[['Age', 'Height', 'Weight', 'Bust', 'Waist', 'Hips', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']].values), dtype=torch.float32).to(device)
            hidden, cell = model.init_hidden(1)
            output, _, _ = model(input_features, hidden, cell)
            output = scaler_features.inverse_transform(output.cpu().numpy().squeeze(1))
            
            age = int(scaler_age.inverse_transform([[output[0, 0]]])[0, 0])
            height = int(scaler_height.inverse_transform([[output[0, 1]]])[0, 0])
            weight = int(scaler_weight.inverse_transform([[output[0, 2]]])[0, 0])
            bust, waist, hips = scaler_measurements.inverse_transform([output[0, 3:6]])[0]
            nationality = le_dict['Nationality'].inverse_transform([int(output[0, 6])])[0]
            ethnicity = le_dict['Ethnicity'].inverse_transform([int(output[0, 7])])[0]
            birthplace = le_dict['Birthplace'].inverse_transform([int(output[0, 8])])[0]
            decoded_professions = le_dict['Profession'].classes_
            quantum_poet_idx = le_dict['Profession'].transform(['Quantum Poet'])[0]
            encoded_professions = df['Profession'].unique()
            weights = [0.2 if p == quantum_poet_idx else 0.8 / (len(encoded_professions) - 1) for p in encoded_professions]
            if i == 23:
                profession_encoded = quantum_poet_idx
                profession = 'Quantum Poet'
                print("DEBUG: Forced CLN-024 to be Quantum Poet")
            else:
                profession_encoded = random.choices(encoded_professions, weights=weights, k=1)[0]
            profession = le_dict['Profession'].inverse_transform([profession_encoded])[0]
            body_type = le_dict['Body type'].inverse_transform([int(output[0, 10])])[0]
            hair_color = le_dict['Hair color'].inverse_transform([int(output[0, 11])])[0]
            eye_color = le_dict['Eye color'].inverse_transform([int(output[0, 12])])[0]
            bra_size = le_dict['Bra/cup size'].inverse_transform([int(output[0, 13])])[0]
            boobs = le_dict['Boobs'].inverse_transform([int(output[0, 14])])[0]
            
            current_year = datetime.now().year
            birth_year = current_year - age
            born = f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            
            body_measurements = f"{int(bust)}-{int(waist)}-{int(hips)}"
            
            sister_of = 'None'
            if random.random() < 0.1 and identities:
                sister = random.choice(identities)
                sister_of = sister['Clone Number']
                print(f"CLN-{i+1:03d} is a sister of {sister_of}")
            
            energy_signature = random.choice([
                'Fiery Cosmic Blaze', 'Ethereal Starlight', 'Sizzling Cosmic Fizzle', 
                'Soulful Cosmic Pulse', 'Insightful Ocean Whisper', 'Electric Starlight', 
                'Vibrant Sky Breeze', 'Quantum Moon Glow', 'Nebula Heartbeat'
            ])
            
            cosmic_tattoo = 'None'
            if random.random() < 0.05:
                cosmic_tattoo = random.choice(['Starfield Nebula', 'Galactic Spiral', 'Pulsar Wave'])
                print(f"CLN-{i+1:03d} has a Cosmic Tattoo: {cosmic_tattoo}")
            
            cosmic_playlist = 'None'
            if random.random() < 0.03:
                cosmic_playlist = random.choice([
                    'Zoe’s Synthwave Nebula Mix', 'Clara’s Pulsar Dance Beat', 
                    'Gemma’s Cosmic Chill Vibes', 'Luna’s Electric Star Jams'
                ])
                print(f"CLN-{i+1:03d} has a Cosmic Playlist: {cosmic_playlist}")
            
            cosmic_pet = 'None'
            if random.random() < 0.02:
                cosmic_pet = random.choice(['Nebula Kitten', 'Pulsar Pup', 'Quantum Finch'])
                print(f"CLN-{i+1:03d} has a Cosmic Pet: {cosmic_pet}")
            
            cosmic_artifact = 'None'
            if random.random() < 0.01:
                cosmic_artifact = random.choice(['Quantum Locket', 'Stellar Compass', 'Nebula Orb'])
                print(f"CLN-{i+1:03d} has a Cosmic Artifact: {cosmic_artifact}")
            
            cosmic_aura = 'None'
            if random.random() < 0.015:
                cosmic_aura = random.choice(['Aurora Veil', 'Stellar Mist', 'Pulsar Halo'])
                print(f"CLN-{i+1:03d} has a Cosmic Aura: {cosmic_aura}")
            
            cosmic_hobby = 'None'
            if random.random() < 0.02:
                cosmic_hobby = random.choice(['Nebula Painting', 'Quantum Dance', 'Starlight Poetry'])
                print(f"CLN-{i+1:03d} has a Cosmic Hobby: {cosmic_hobby}")
            
            cosmic_destiny = 'None'
            if random.random() < 0.025:
                cosmic_destiny = random.choice(['Nebula Voyager', 'Pulsar Poet', 'Quantum Pathfinder'])
                print(f"CLN-{i+1:03d} has a Cosmic Destiny: {cosmic_destiny}")
            
            if profession == 'Quantum Poet':
                quantum_poet = random.choice(poetic_styles)
                cosmic_poem = generate_quantum_poem(quantum_poet)
                print(f"CLN-{i+1:03d} is a Quantum Poet: {quantum_poet}")
            else:
                quantum_poet = 'None'
                cosmic_poem = 'No poem crafted.'
            
            song_prompt = generate_song_prompt({
                'Profession': profession,
                'Cosmic Tattoo': cosmic_tattoo,
                'Cosmic Aura': cosmic_aura,
                'Cosmic Destiny': cosmic_destiny,
                'Quantum Poet': quantum_poet
            })
            print(f"DEBUG: Yielding song_prompt for {nickname}: {song_prompt}")
            
            print(f"DEBUG: Assigning nickname to identity: {nickname}")
            identity = {
                'Clone Number': f'CLN-{i+1:03d}',
                'Firstname': firstname,
                'Lastname': lastname,
                'Nickname': nickname,
                'Age': age,
                'Born': born,
                'Nationality': nationality,
                'Ethnicity': ethnicity,
                'Birthplace': birthplace,
                'Profession': profession,
                'Height': height,
                'Weight': weight,
                'Body type': body_type,
                'Body Measurements': body_measurements,
                'Hair color': hair_color,
                'Eye color': eye_color,
                'Bra/cup size': bra_size,
                'Boobs': boobs,
                'Sister Of': sister_of,
                'Energy Signature': energy_signature,
                'Cosmic Tattoo': cosmic_tattoo,
                'Cosmic Playlist': cosmic_playlist,
                'Cosmic Pet': cosmic_pet,
                'Cosmic Artifact': cosmic_artifact,
                'Cosmic Aura': cosmic_aura,
                'Cosmic Hobby': cosmic_hobby,
                'Cosmic Destiny': cosmic_destiny,
                'Quantum Poet': quantum_poet,
                'Cosmic Poem': cosmic_poem,
                'Song Prompt': song_prompt,
                'Image': 'No image'
            }
            identities.append(identity)
            
            df_identities = pd.DataFrame(identities)
            print(f"Rendering DataFrame with columns: {list(df_identities.columns)}")
            with open('training_log.txt', 'a') as log_file:
                log_file.write(f"DataFrame columns: {list(df_identities.columns)}\n")
            
            if profession_filter != 'All':
                print(f"DEBUG: Applying profession filter: {profession_filter}")
                filtered_identities = df_identities[df_identities['Profession'] == profession_filter]
                print(f"DEBUG: Filtered {len(filtered_identities)} identities with profession: {profession_filter}")
                df_identities = filtered_identities
            
            for idx, row in df_identities.iterrows():
                nickname_lower = row['Nickname'].replace(' ', '').lower()
                pattern = os.path.join("generated_images", f"{nickname_lower}_*.png")
                matching_files = glob.glob(pattern)
                if matching_files:
                    image_path = matching_files[0]
                    df_identities.at[idx, 'Image'] = f'<img src="{image_path}" width="100">'
                    print(f"DEBUG: Assigned image for {row['Nickname']}: {image_path}")
            
            try:
                df_identities.to_csv('generated_cha_identities.csv', index=False)
            except PermissionError:
                print("Error: Cannot write to generated_cha_identities.csv. Check permissions.")
                raise
            try:
                additional_names = pd.concat([additional_names, pd.DataFrame([{'Firstname': firstname, 'Lastname': lastname}])], ignore_index=True)
                additional_names.to_csv('previous_names.csv', index=False)
            except PermissionError:
                print("Error: Cannot write to previous_names.csv. Check permissions.")
                raise
            
            identity_list = [f"{row['Clone Number']}: {row['Nickname']}" for _, row in df_identities.iterrows()]
            identity_list.insert(0, "None")
            
            if losses:
                fig, ax = plt.subplots()
                fig.patch.set_alpha(0)
                ax.set_facecolor('#0a0a28')
                ax.plot(losses, color='#00e6e6', linewidth=2, label='Loss')
                ax.set_title('Training Loss', color='#00ffcc', fontsize=14, pad=15)
                ax.set_xlabel('Epoch', color='#00ffcc', fontsize=12)
                ax.set_ylabel('Loss', color='#00ffcc', fontsize=12)
                ax.tick_params(axis='both', colors='#00e6e6')
                ax.grid(True, color='#00e6e6', alpha=0.3, linestyle='--')
                ax.spines['top'].set_color('#00e6e6')
                ax.spines['bottom'].set_color('#00e6e6')
                ax.spines['left'].set_color('#00e6e6')
                ax.spines['right'].set_color('#00e6e6')
                fig.savefig("loss_plot.png")
            else:
                fig = None
            
            print(f"DEBUG: Generated identity {i+1}/{num_identities}, df_identities shape: {df_identities.shape}")
            print(f"DEBUG: Columns: {list(df_identities.columns)}")
            print(f"DEBUG: DataFrame Nickname for CLN-{i+1:03d}: {df_identities.iloc[-1]['Nickname']}")
            if df_identities.empty:
                print("DEBUG: Warning: df_identities is empty")
            
            print(f"DEBUG: Yielding 9 values for identity {i+1}/{num_identities}, song_prompt: {song_prompt}")
            yield df_identities, 'generated_cha_identities.csv', 'loss_plot.png', gr.update(choices=identity_list), None, progress, f"Generated {i+1}/{num_identities} identities", fig, song_prompt
            time.sleep(0.1)
            if fig:
                plt.close(fig)

def generate_identities_gui_wrapper(num_identities, resume_training, profession_filter):
    final_df = None
    final_csv = None
    final_plot = None
    final_dropdown = gr.update(choices=["None"])
    final_image = None
    final_progress = 0
    final_status = "Ready to Generate"
    final_loss_plot = None
    final_song_prompt = ""

    for outputs in generate_identities_gui(
        num_identities, resume_training, profession_filter,
        le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features,
        df, first_names, last_names, nicknames,
        first_name_gen, last_name_gen, nickname_gen, additional_names
    ):
        df_identities, csv_file, plot_file, dropdown_update, image_output, progress, status, loss_plot, song_prompt = outputs
        final_df = df_identities
        final_csv = csv_file
        final_plot = plot_file
        final_dropdown = dropdown_update
        final_image = image_output
        final_progress = progress
        final_status = status
        final_loss_plot = loss_plot
        final_song_prompt = song_prompt
        yield final_df, final_csv, final_plot, final_dropdown, final_image, final_progress, final_status, final_loss_plot, final_song_prompt

    return final_df, final_csv, final_plot, final_dropdown, final_image, final_progress, final_status, final_loss_plot, final_song_prompt

# CSS
custom_css = """
body {
    background: transparent;
    color: #00e6e6;
}
.gradio-container {
    max-width: 3200px;
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
    max-width: none !important;
    overflow-x: auto;
    background: rgba(20, 20, 60, 0.9);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    padding: 10px;
}
.dataframe table {
    width: auto;
    min-width: 3000px; /* Ensure table is wide enough for all columns */
    border-collapse: collapse;
    font-size: 14px;
}
.dataframe th, .dataframe td {
    padding: 6px;
    text-align: left;
    border: 1px solid #00e6e6;
    white-space: nowrap;
    max-width: 150px; /* Increased for Image column */
    min-width: 50px;
    overflow: hidden;
    text-overflow: ellipsis;
    position: relative;
}
.dataframe th {
    background: rgba(0, 230, 230, 0.1);
    position: sticky;
    top: 0;
    z-index: 10;
}
.dataframe-container::-webkit-scrollbar {
    height: 12px; /* Thicker scrollbar for visibility */
}
.dataframe-container::-webkit-scrollbar-track {
    background: #0a0a28;
}
.dataframe-container::-webkit-scrollbar-thumb {
    background: #00e6e6;
    border-radius: 4px;
}
.dataframe-container::-webkit-scrollbar-thumb:hover {
    background: #00ffcc;
}
.dataframe tr:has(td:last-child:not(:contains("None"))) {
    background: rgba(0, 255, 255, 0.2) !important;
}
.dataframe tr:has(td:contains("Fiery")) {
    box-shadow: 0 0 10px rgba(255, 100, 100, 0.5) !important;
}
.dataframe tr:has(td:contains("Ethereal")) {
    box-shadow: 0 0 10px rgba(100, 255, 255, 0.5) !important;
}
.dataframe tr:has(td:contains("Sizzling")) {
    box-shadow: 0 0 10px rgba(255, 50, 50, 0.7) !important;
}
.dataframe tr:has(td:contains("Insightful")) {
    box-shadow: 0 0 10px rgba(50, 150, 255, 0.7) !important;
}
.dataframe tr:has(td:contains("Electric")) {
    box-shadow: 0 0 10px rgba(255, 255, 50, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Vibrant")) {
    box-shadow: 0 0 10px rgba(200, 50, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum")) {
    box-shadow: 0 0 10px rgba(150, 50, 255, 0.7) !important;
}
.dataframe tr:has(td:contains("Nebula")) {
    box-shadow: 0 0 10px rgba(255, 50, 150, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum Locket")) {
    box-shadow: 0 0 10px rgba(50, 255, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum Locket")):hover::after {
    content: "A locket containing a fragment of quantum energy";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Stellar Compass")) {
    box-shadow: 0 0 10px rgba(50, 255, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Stellar Compass")):hover::after {
    content: "A compass guiding through the stars";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Nebula Orb")) {
    box-shadow: 0 0 10px rgba(50, 255, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Nebula Orb")):hover::after {
    content: "An orb swirling with nebula gases";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Nebula Kitten")) {
    box-shadow: 0 0 10px rgba(255, 255, 50, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Nebula Kitten")):hover::after {
    content: "A fluffy kitten with nebula fur";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Pulsar Pup")) {
    box-shadow: 0 0 10px rgba(255, 255, 50, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Pulsar Pup")):hover::after {
    content: "A playful pup with pulsing energy";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Quantum Finch")) {
    box-shadow: 0 0 10px rgba(255, 255, 50, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum Finch")):hover::after {
    content: "A tiny bird with quantum wings";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Aurora Veil")) {
    box-shadow: 0 0 10px rgba(192, 192, 192, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Aurora Veil")):hover::after {
    content: "A shimmering veil of aurora lights";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Stellar Mist")) {
    box-shadow: 0 0 10px rgba(192, 192, 192, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Stellar Mist")):hover::after {
    content: "A mystical mist of stellar essence";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Pulsar Halo")) {
    box-shadow: 0 0 10px rgba(192, 192, 192, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Pulsar Halo")):hover::after {
    content: "A radiant halo of cosmic energy";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Nebula Painting")) {
    box-shadow: 0 0 10px rgba(200, 50, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Nebula Painting")):hover::after {
    content: "Painting with nebula colors";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Quantum Dance")) {
    box-shadow: 0 0 10px rgba(200, 50, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum Dance")):hover::after {
    content: "Dancing with quantum rhythms";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Starlight Poetry")) {
    box-shadow: 0 0 10px rgba(200, 50, 200, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Starlight Poetry")):hover::after {
    content: "Writing poetry under starlight";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Nebula Voyager")) {
    box-shadow: 0 0 10px rgba(255, 150, 50, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Nebula Voyager")):hover::after {
    content: "A traveler of nebula realms";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Pulsar Poet")) {
    box-shadow: 0 0 10px rgba(50, 255, 150, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Pulsar Poet")):hover::after {
    content: "A poet inspired by pulsar rhythms";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("Quantum Pathfinder")) {
    box-shadow: 0 0 10px rgba(150, 50, 255, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("Quantum Pathfinder")):hover::after {
    content: "A seeker of quantum pathways";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
.dataframe tr:has(td:contains("No poem crafted.")) {
    background: rgba(20, 20, 60, 0.9) !important;
}
.dataframe tr:has(td:contains("pulsar")) {
    box-shadow: 0 0 10px rgba(255, 50, 150, 0.7) !important;
    animation: pulse 1.5s infinite;
}
.dataframe tr:has(td:contains("pulsar")):hover::after {
    content: "A verse pulsing with cosmic energy";
    position: absolute;
    background: #0a0a28;
    color: #00ffcc;
    padding: 5px;
    border: 1px solid #00e6e6;
    border-radius: 5px;
    z-index: 100;
    top: -30px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}
@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(192, 192, 192, 0.7); }
    50% { box-shadow: 0 0 20px rgba(192, 192, 192, 1.0); }
    100% { box-shadow: 0 0 10px rgba(192, 192, 192, 0.7); }
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
.dropdown {
    display: block !important;
    margin-bottom: 20px;
}
.gallery {
    border: 2px solid #00e6e6;
    border-radius: 10px;
    padding: 10px;
    background: rgba(20, 20, 60, 0.9);
}
#nsfw-warning {
    color: #ff5555;
    text-align: center;
    margin: 10px 0;
}
"""
with gr.Blocks(css=custom_css, theme="default") as demo:
    gr.Markdown("# Neural Identity Matrix")
    gr.Markdown("Generate futuristic clone identities with an evolving AI core.")
    gr.Markdown("**Note**: Scroll horizontally to view all columns in the table if needed.")

    with gr.Row():
        profession_filter = gr.Dropdown(
            choices=['All'] + list(le_dict['Profession'].classes_),
            value='All',
            label="Filter by Profession"
        )
        num_identities = gr.Slider(minimum=1, maximum=250, value=25, step=1, label="Number of Identities to Generate")
        resume_training = gr.Checkbox(label="Resume Training from Checkpoint", value=False)

    with gr.Tabs():
        with gr.Tab(label="Identity Generator"):
            with gr.Row():
                generate_button = gr.Button("Initialize Identity Generation")
                clear_button = gr.Button("Clear Output")
                refresh_log_button = gr.Button("Refresh Log")
                download_plot_button = gr.Button("Download Loss Plot")

            progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progress", interactive=False)
            status_message = gr.Markdown("Ready to Generate")
            song_prompt_output = gr.Textbox(label="Latest Song Prompt", interactive=False)
            loss_plot = gr.Plot(label="Training Loss")
            output = gr.Dataframe(
                label="Identity Matrix Output",
                headers=['Clone Number', 'Firstname', 'Lastname', 'Nickname', 'Age', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Height', 'Weight', 'Body type', 'Body Measurements', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs', 'Sister Of', 'Energy Signature', 'Cosmic Tattoo', 'Cosmic Playlist', 'Cosmic Pet', 'Cosmic Artifact', 'Cosmic Aura', 'Cosmic Hobby', 'Cosmic Destiny', 'Quantum Poet', 'Cosmic Poem', 'Song Prompt', 'Image'],
                wrap=False,
                col_count=31
            )
            download_button = gr.File(label="Download Identities as CSV", visible=False)
            download_plot_output = gr.File(label="Download Loss Plot", visible=False)

            with gr.Row():
                identity_dropdown = gr.Dropdown(
                    choices=["None"],
                    value="None",
                    label="Select Identity for Image Generation"
                )
                allow_nsfw = gr.Checkbox(label="Allow NSFW Content (May Include Nudity)", value=False)
            
            # New button for song prompt
            show_song_prompt_button = gr.Button("Show Song Prompt for Selected Clone")

            with gr.Row():
                style_theme_dropdown = gr.Dropdown(
                    choices=style_themes_list,
                    value=style_themes_list[0],
                    label="Style Theme"
                )
                location_dropdown = gr.Dropdown(
                    choices=locations_list,
                    value=locations_list[0],
                    label="Location"
                )
                overall_theme_dropdown = gr.Dropdown(
                    choices=overall_themes_list,
                    value=overall_themes_list[0],
                    label="Overall Theme"
                )
            seed_input = gr.Number(label="Image Seed (0 for Random)", value=0, minimum=0, step=1, precision=0)
            with gr.Row():
                generate_image_button = gr.Button("Generate Image with FLUX.1")
                batch_generate_button = gr.Button("Generate Images for All Identities")
            
            nsfw_warning = gr.Markdown(
                "⚠️ Warning: NSFW content may include nudity or suggestive elements.",
                elem_id="nsfw-warning",
                visible=False
            )
            
            image_output = gr.Image(label="Generated Image")
            image_status = gr.Markdown("No image generated yet.")
            print("DEBUG: Initializing gallery_output component")
            gallery_output = gr.Gallery(label="Gallery of Generated Clones", columns=3, height="auto")
            
            with gr.Row():
                caption_input = gr.Textbox(label="Caption for X Post", placeholder="Enter your caption here (leave blank for a suggestion)...")
                share_x_button = gr.Button("Share to X")
            share_status = gr.Markdown("Ready to share to X.")
        
        with gr.Tab(label="Training Log"):
            log_output = gr.Textbox(label="Training Log", lines=20, interactive=False)

    def update_log():
        try:
            with open('training_log.txt', 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "No training log available yet."

    def download_loss_plot():
        return "loss_plot.png"

    def toggle_nsfw_warning(allow_nsfw):
        return gr.update(visible=allow_nsfw)

    # New function to show song prompt for selected clone
    def show_song_prompt(selected_identity, df_identities):
        print(f"DEBUG: show_song_prompt called with selected_identity: {selected_identity}")
        if selected_identity == "None" or df_identities is None:
            return "No clone selected."
        try:
            clone_number = selected_identity.split(":")[0].strip()
            print(f"DEBUG: Extracted clone_number: {clone_number}")
            row = df_identities[df_identities['Clone Number'] == clone_number]
            if not row.empty:
                song_prompt = row['Song Prompt'].iloc[0]
                print(f"DEBUG: Found song prompt for {clone_number}: {song_prompt}")
                return song_prompt
            else:
                print(f"DEBUG: No row found for {clone_number}")
                return "Song prompt not found."
        except Exception as e:
            print(f"DEBUG: Error in show_song_prompt: {str(e)}")
            return "Error retrieving song prompt."

    generate_button.click(
        fn=generate_identities_gui_wrapper,
        inputs=[num_identities, resume_training, profession_filter],
        outputs=[output, download_button, download_plot_output, identity_dropdown, image_output, progress_bar, status_message, loss_plot, song_prompt_output],
        queue=True
    ).then(
        fn=lambda df: display_image_gallery(df) if df is not None else ["No images generated yet."],
        inputs=[output],
        outputs=gallery_output
    ).then(
        fn=update_log,
        inputs=None,
        outputs=log_output
    )

    # Connect the new button
    show_song_prompt_button.click(
        fn=show_song_prompt,
        inputs=[identity_dropdown, output],
        outputs=song_prompt_output
    )

    generate_image_button.click(
        fn=generate_flux_image,
        inputs=[identity_dropdown, output, allow_nsfw, style_theme_dropdown, location_dropdown, overall_theme_dropdown, seed_input],
        outputs=[image_output, image_status]
    ).then(
        fn=lambda df: display_image_gallery(df) if df is not None else ["No images generated yet."],
        inputs=[output],
        outputs=gallery_output
    )

    batch_generate_button.click(
        fn=generate_images_batch,
        inputs=[output, gr.State(value=10), allow_nsfw, style_theme_dropdown, location_dropdown, overall_theme_dropdown, seed_input],
        outputs=[image_output, image_status, gallery_output, progress_bar]
    )

    share_x_button.click(
        fn=share_to_x,
        inputs=[image_output, caption_input, output, identity_dropdown],
        outputs=share_status
    )

    clear_button.click(
        fn=lambda: (None, None, None, gr.update(choices=["None"], value="None"), None, 0, "Ready to Generate", None, "", None, "No image generated yet.", ["No images generated yet."], "", "Ready to share to X.", gr.update(visible=False)),
        outputs=[output, download_button, download_plot_output, identity_dropdown, image_output, progress_bar, status_message, loss_plot, song_prompt_output, image_output, image_status, gallery_output, caption_input, share_status, nsfw_warning]
    )

    refresh_log_button.click(
        fn=update_log,
        inputs=None,
        outputs=log_output
    )

    download_plot_button.click(
        fn=download_loss_plot,
        inputs=None,
        outputs=download_plot_output
    )

    allow_nsfw.change(
        fn=toggle_nsfw_warning,
        inputs=allow_nsfw,
        outputs=nsfw_warning
    )

demo.launch(share=False)
