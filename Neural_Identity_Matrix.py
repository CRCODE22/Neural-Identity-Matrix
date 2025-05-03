import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import gradio as gr
import pickle
import os
from tqdm import tqdm  # For console progress bar

# Step 1: Generate Initial Synthetic Dataset
first_names = ["Emma", "Olivia", "Sophia", "Ava", "Isabella", "Mia", "Charlotte", "Amelia"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
nicknames = ["Em", "Liv", "Sophie", "Izzy", "Bella", "Mimi", "Charlie", "Amy"]
birthplaces = ["New York, USA", "London, UK", "Tokyo, Japan", "Sydney, Australia", "Paris, France"]
nationalities = ["American", "British", "Japanese", "Australian", "French"]
ethnicities = ["Caucasian", "African", "Asian", "Hispanic", "Mixed"]
professions = ["Engineer", "Artist", "Doctor", "Teacher", "Designer"]
hair_colors = ["Blonde", "Brown", "Black", "Red"]
eye_colors = ["Blue", "Brown", "Green", "Hazel"]
body_types = ["Athletic", "Slim", "Curvy"]
bra_sizes = ["A", "B", "C", "D"]
boob_types = ["Natural", "Enhanced"]

current_date = datetime(2025, 5, 3)

def random_birth_date(age):
    start_date = current_date - timedelta(days=(age + 1) * 365)
    end_date = current_date - timedelta(days=age * 365)
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

def generate_initial_dataset(n_samples=1000):
    data = []
    for _ in range(n_samples):
        age = random.randint(18, 40)
        height = random.randint(150, 180)
        weight = random.randint(45, 80)
        bust = random.randint(80, 100)
        waist = random.randint(60, 80)
        hips = random.randint(80, 100)
        profile = {
            "Firstname": random.choice(first_names),
            "Lastname": random.choice(last_names),
            "Nickname": random.choice(nicknames),
            "Age": age,
            "Born": random_birth_date(age).strftime("%Y-%m-%d"),
            "Birthplace": random.choice(birthplaces),
            "Nationality": random.choice(nationalities),
            "Ethnicity": random.choice(ethnicities),
            "Profession": random.choice(professions),
            "Hair color": random.choice(hair_colors),
            "Eye color": random.choice(eye_colors),
            "Height": height,
            "Weight": weight,
            "Body type": random.choice(body_types),
            "Body Measurements": f"{bust}-{waist}-{hips}",
            "Bra/cup size": random.choice(bra_sizes),
            "Boobs": random.choice(boob_types)
        }
        data.append(profile)
    return pd.DataFrame(data)

df = generate_initial_dataset()

# Step 2: Data Preprocessing
le_dict = {field: LabelEncoder() for field in [
    "Hair color", "Eye color", "Body type", "Bra/cup size", "Boobs",
    "Firstname", "Lastname", "Nickname", "Birthplace", "Nationality", "Ethnicity", "Profession"
]}
for field in le_dict:
    le_dict[field].fit(df[field])

scaler_age = MinMaxScaler()
scaler_age.fit(df[["Age"]])

scaler_height = MinMaxScaler()
scaler_height.fit(df[["Height"]])

scaler_weight = MinMaxScaler()
scaler_weight.fit(df[["Weight"]])

df_measurements = df["Body Measurements"].str.split("-", expand=True).astype(int)
df_measurements.columns = ["Bust", "Waist", "Hips"]
scaler_measurements = MinMaxScaler()
scaler_measurements.fit(df_measurements)

class CloneIdentityDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.reference_date = datetime(1985, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = {}
        for field in le_dict:
            encoded[field] = torch.tensor(le_dict[field].transform([row[field]])[0], dtype=torch.long)
        
        encoded["Age"] = torch.tensor(scaler_age.transform(pd.DataFrame([[row["Age"]]], columns=["Age"]))[0], dtype=torch.float)
        encoded["Height"] = torch.tensor(scaler_height.transform(pd.DataFrame([[row["Height"]]], columns=["Height"]))[0], dtype=torch.float)
        encoded["Weight"] = torch.tensor(scaler_weight.transform(pd.DataFrame([[row["Weight"]]], columns=["Weight"]))[0], dtype=torch.float)
        
        measurements = [float(x) for x in row["Body Measurements"].split("-")]
        measurements_df = pd.DataFrame([measurements], columns=["Bust", "Waist", "Hips"])
        encoded["Body Measurements"] = torch.tensor(scaler_measurements.transform(measurements_df)[0], dtype=torch.float)
        
        born = datetime.strptime(row["Born"], "%Y-%m-%d")
        days_since_ref = (born - self.reference_date).days
        encoded["Born"] = torch.tensor([days_since_ref], dtype=torch.float)
        return encoded

# Step 3: Neural Network
class IdentityGenerator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256):
        super(IdentityGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.heads = nn.ModuleDict({
            field: nn.Linear(hidden_dim, len(le_dict[field].classes_))
            for field in le_dict
        })
        self.heads["Age"] = nn.Linear(hidden_dim, 1)
        self.heads["Height"] = nn.Linear(hidden_dim, 1)
        self.heads["Weight"] = nn.Linear(hidden_dim, 1)
        self.heads["Body Measurements"] = nn.Linear(hidden_dim, 3)
        self.heads["Born"] = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        outputs = {field: self.heads[field](features) for field in self.heads}
        return outputs

# Step 4: Quality Filter
def filter_identity(identity):
    measurements = [int(x) for x in identity["Body Measurements"].split("-")]
    if not (70 <= measurements[0] <= 110 and 50 <= measurements[1] <= 90 and 70 <= measurements[2] <= 110):
        return False
    if not (140 <= identity["Height"] <= 190 and 35 <= identity["Weight"] <= 90):
        return False
    born = datetime.strptime(identity["Born"], "%Y-%m-%d")
    age = (current_date - born).days // 365
    if not (15 <= age <= 50):
        return False
    return True

# Step 5: Generate Identity with Probabilistic Sampling
def generate_identity(model, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, model.input_dim).to(device)
        outputs = model(x)
        identity = {}
        for field in le_dict:
            probs = torch.softmax(outputs[field], dim=1)
            idx = torch.multinomial(probs, 1).item()
            identity[field] = le_dict[field].inverse_transform([idx])[0]
        identity["Age"] = scaler_age.inverse_transform(outputs["Age"].cpu().numpy())[0][0]
        identity["Height"] = scaler_height.inverse_transform(outputs["Height"].cpu().numpy())[0][0]
        identity["Weight"] = scaler_weight.inverse_transform(outputs["Weight"].cpu().numpy())[0][0]
        measurements = scaler_measurements.inverse_transform(outputs["Body Measurements"].cpu().numpy())
        identity["Body Measurements"] = f"{int(measurements[0][0])}-{int(measurements[0][1])}-{int(measurements[0][2])}"
        days_since_ref = outputs["Born"].item()
        min_days = (datetime(1980, 1, 1) - datetime(1985, 1, 1)).days
        max_days = (datetime(2007, 1, 1) - datetime(1985, 1, 1)).days
        days_since_ref = min(max(days_since_ref, min_days), max_days)
        born_date = datetime(1985, 1, 1) + timedelta(days=days_since_ref)
        identity["Born"] = born_date.strftime("%Y-%m-%d")
    return identity

# Step 6: Save and Load Model, Optimizer, and Preprocessors
def save_training_state(model, optimizer, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, df, path_prefix="checkpoint"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{path_prefix}_model_optimizer.pth")
    
    with open(f"{path_prefix}_preprocessors.pkl", "wb") as f:
        pickle.dump({
            'le_dict': le_dict,
            'scaler_age': scaler_age,
            'scaler_height': scaler_height,
            'scaler_weight': scaler_weight,
            'scaler_measurements': scaler_measurements
        }, f)
    
    df.to_csv(f"{path_prefix}_learned_identities.csv", index=False)
    print(f"Saved training state to {path_prefix}_* files.")

def load_training_state(model, optimizer, path_prefix="checkpoint"):
    global le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, df
    checkpoint = torch.load(f"{path_prefix}_model_optimizer.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    with open(f"{path_prefix}_preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
        le_dict = preprocessors['le_dict']
        scaler_age = preprocessors['scaler_age']
        scaler_height = preprocessors['scaler_height']
        scaler_weight = preprocessors['scaler_weight']
        scaler_measurements = preprocessors['scaler_measurements']
    
    df = pd.read_csv(f"{path_prefix}_learned_identities.csv")
    print(f"Loaded training state from {path_prefix}_* files.")
    return model, optimizer

# Step 7: Train Model
def train_model(resume=False):
    global df
    dataset = CloneIdentityDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = IdentityGenerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if resume and os.path.exists("checkpoint_model_optimizer.pth"):
        model, optimizer = load_training_state(model, optimizer)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    n_iterations = 2
    n_epochs_per_cycle = 10

    total_steps = n_iterations * n_epochs_per_cycle * len(dataloader)
    step = 0

    for iteration in range(n_iterations):
        print(f"Self-Improvement Cycle {iteration + 1}")
        with tqdm(total=len(dataloader), desc=f"Cycle {iteration + 1}, Epoch", unit="batch") as pbar:
            for epoch in range(n_epochs_per_cycle):
                total_loss = 0
                for batch_idx, batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    batch_size = len(batch["Firstname"])
                    x = torch.randn(batch_size, model.input_dim).to(device)
                    outputs = model(x)

                    loss = 0
                    for field in le_dict:
                        loss += criterion_ce(outputs[field], batch[field].to(device))
                    for field in ["Age", "Height", "Weight", "Born"]:
                        loss += criterion_mse(outputs[field], batch[field].to(device))
                    loss += criterion_mse(outputs["Body Measurements"], batch["Body Measurements"].to(device))

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    step += 1
                    pbar.update(1)
                    progress_value = (step / total_steps) * 50  # 0-50% for training
                    status = f"Training Model (Cycle {iteration+1}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)})"
                    yield None, None, progress_value, status

                print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        new_data = []
        for i in range(100):
            progress_value = 50 + (i / 100) * 25  # 50-75% for generating new identities
            status = f"Generating New Identities for Dataset ({i+1}/100)"
            yield None, None, progress_value, status
            identity = generate_identity(model, device)
            if filter_identity(identity):
                new_data.append(identity)

        if new_data:
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            print(f"Added {len(new_data)} new identities. Total dataset size: {len(df)}")
            for field in le_dict:
                le_dict[field].fit(df[field])
            dataset = CloneIdentityDataset(df)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        save_training_state(model, optimizer, le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, df)

    return model, device

# Step 8: Gradio Interface with Enhanced Futuristic Features
custom_css = """
body {
    background: linear-gradient(135deg, #0d0d2b 0%, #1a1a4d 100%);
    color: #00e6e6;
    font-family: 'Orbitron', 'Arial', sans-serif;
    position: relative;
    overflow: hidden;
}
#matrix-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    opacity: 0.3;
}
.gradio-container {
    max-width: 1200px;
    margin: auto;
    border: 2px solid #00e6e6;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0, 230, 230, 0.5);
    background: rgba(10, 10, 40, 0.8);
    padding: 20px;
    position: relative;
    z-index: 1;
}
h1, h2, h3 {
    text-align: center;
    color: #00ffcc;
    text-shadow: 0 0 10px #00ffcc;
}
button {
    background: #1a1a4d;
    color: #00e6e6;
    border: 2px solid #00e6e6;
    border-radius: 10px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    animation: pulse 2s infinite;
}
button:hover {
    background: #00e6e6;
    color: #0d0d2b;
    box-shadow: 0 0 15px #00e6e6;
}
@keyframes pulse {
    0% { box-shadow: 0 0 5px #00e6e6; }
    50% { box-shadow: 0 0 20px #00e6e6; }
    100% { box-shadow: 0 0 5px #00e6e6; }
}
.dataframe-container {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto;
    background: rgba(20, 20, 60, 0.9);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    animation: fadeIn 1s ease-in;
}
.dataframe table {
    width: 100%;
    min-width: 1200px;
    border-collapse: collapse;
}
.dataframe th, .dataframe td {
    padding: 8px;
    text-align: left;
    border: 1px solid #00e6e6;
    white-space: nowrap;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
#neural-activity {
    margin-top: 20px;
    height: 100px;
    background: rgba(10, 10, 40, 0.8);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    position: relative;
    overflow: hidden;
}
.neural-node {
    position: absolute;
    background: #00ffcc;
    border-radius: 50%;
    width: 10px;
    height: 10px;
    animation: moveNode 5s infinite linear;
}
@keyframes moveNode {
    0% { transform: translate(0, 0); opacity: 0.8; }
    50% { opacity: 0.3; }
    100% { transform: translate(1000px, 100px); opacity: 0.8; }
}
#neural-network-3d {
    margin-top: 20px;
    height: 300px;
    background: rgba(10, 10, 40, 0.8);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    position: relative;
    overflow: hidden;
}
#neural-network-3d canvas {
    width: 100% !important;
    height: 100% !important;
}
#visualization-error {
    text-align: center;
    color: #ff5555;
    padding: 10px;
}
#loader {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 50px;
    height: 50px;
    border: 5px solid #00e6e6;
    border-top: 5px solid transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    z-index: 1000;
}
@keyframes spin {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}
.slider-container {
    margin: 10px 0;
}
.slider-container .slider {
    background: rgba(10, 10, 40, 0.8);
    border: 2px solid #00e6e6;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 230, 230, 0.5);
}
.slider-container .slider::-webkit-slider-thumb {
    background: linear-gradient(90deg, #00e6e6, #00ffcc);
    border: 2px solid #00ffcc;
    box-shadow: 0 0 10px #00ffcc;
    animation: glow 2s infinite;
}
.slider-container .slider::-moz-range-thumb {
    background: linear-gradient(90deg, #00e6e6, #00ffcc);
    border: 2px solid #00ffcc;
    box-shadow: 0 0 10px #00ffcc;
    animation: glow 2s infinite;
}
@keyframes glow {
    0% { box-shadow: 0 0 5px #00e6e6; }
    50% { box-shadow: 0 0 20px #00ffcc; }
    100% { box-shadow: 0 0 5px #00e6e6; }
}
#status-message {
    text-align: center;
    color: #00ffcc;
    text-shadow: 0 0 5px #00ffcc;
    margin: 10px 0;
}
"""

def generate_identities_gui(num_identities, resume_training):
    print(f"Starting identity generation: num_identities={num_identities}, resume_training={resume_training}")
    yield None, None, 0, "Starting Training..."
    model, device = yield from train_model(resume=resume_training)

    identities = []
    num_identities = int(num_identities)
    for i in range(num_identities):
        progress_value = 50 + (i / num_identities) * 25  # 50-75% for generating identities
        status = f"Generating Identity {i+1}/{num_identities}"
        yield None, None, progress_value, status
        identity = generate_identity(model, device)
        print(f"Generated Identity {i+1}: {identity}")
        identities.append(identity)

    filtered_identities = []
    for i, identity in enumerate(identities):
        progress_value = 75 + (i / len(identities)) * 25  # 75-100% for filtering
        status = f"Filtering Identity {i+1}/{len(identities)}"
        yield None, None, progress_value, status
        if filter_identity(identity):
            filtered_identities.append(identity)

    print(f"Generated {num_identities} identities, {len(filtered_identities)} passed the filter.")
    if filtered_identities:
        df_identities = pd.DataFrame(filtered_identities)
        csv_path = "generated_identities.csv"
        df_identities.to_csv(csv_path, index=False)
        yield df_identities, csv_path, 100, "Generation Complete"
    else:
        yield None, None, 100, "Generation Complete"

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script>
        // Dynamically inject CSP meta tag into the head
        (function() {
            console.log("Attempting to inject CSP meta tag...");
            const meta = document.createElement('meta');
            meta.setAttribute('http-equiv', 'Content-Security-Policy');
            meta.setAttribute('content', "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; media-src 'self';");
            const head = document.head || document.getElementsByTagName('head')[0];
            if (head) {
                head.appendChild(meta);
                console.log("CSP meta tag injected successfully!");
            } else {
                console.error("Failed to find <head> element to inject CSP meta tag.");
            }
        })();

        document.addEventListener('DOMContentLoaded', function() {
            console.log("DOM fully loaded - Inline script loaded successfully!");

            // Utility to wait for an element to appear in the DOM
            function waitForElement(id, callback, maxAttempts = 50, interval = 200) {
                let attempts = 0;
                const intervalId = setInterval(() => {
                    const element = document.getElementById(id);
                    if (element) {
                        clearInterval(intervalId);
                        console.log(`Element ${id} found!`);
                        callback(element);
                    } else if (attempts >= maxAttempts) {
                        clearInterval(intervalId);
                        console.error(`Element ${id} not found after ${maxAttempts} attempts.`);
                    }
                    attempts++;
                }, interval);
            }

            function startNeuralAnimation() {
                waitForElement('neural-activity', (container) => {
                    try {
                        console.log("Starting 2D Neural Animation...");
                        container.innerHTML = ''; // Clear existing nodes
                        for (let i = 0; i < 20; i++) {
                            const node = document.createElement('div');
                            node.className = 'neural-node';
                            node.style.left = Math.random() * 100 + '%';
                            node.style.top = Math.random() * 100 + '%';
                            node.style.animationDelay = Math.random() * 5 + 's';
                            container.appendChild(node);
                        }
                        setInterval(() => {
                            container.style.boxShadow = '0 0 ' + (Math.random() * 20 + 10) + 'px #00e6e6';
                        }, 1000);
                    } catch (error) {
                        console.error("Error in startNeuralAnimation:", error);
                    }
                });
            }

            // 3D Neural Network Visualization
            function start3DNeuralNetwork() {
                waitForElement('neural-network-3d', (container) => {
                    try {
                        console.log("Starting 3D Neural Network Visualization...");

                        // Check for WebGL support
                        const canvas = document.createElement('canvas');
                        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                        if (!gl) {
                            container.innerHTML = '<div id="visualization-error">WebGL is not supported in your browser. Please use a modern browser with WebGL enabled.</div>';
                            return;
                        }

                        // Check if THREE is defined
                        if (typeof THREE === 'undefined') {
                            console.error("Three.js failed to load.");
                            container.innerHTML = '<div id="visualization-error">Failed to load Three.js. Please check your internet connection.</div>';
                            return;
                        }

                        const scene = new THREE.Scene();
                        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 300, 0.1, 1000);
                        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                        renderer.setSize(container.clientWidth, 300);
                        container.appendChild(renderer.domElement);

                        // Neural Network Structure: Input (5) -> Hidden (4, 3) -> Output (2)
                        const layers = [
                            { neurons: 5, x: -200 }, // Input layer
                            { neurons: 4, x: -100 }, // Hidden layer 1
                            { neurons: 3, x: 0 },    // Hidden layer 2
                            { neurons: 2, x: 100 }   // Output layer
                        ];

                        const nodes = [];
                        const edges = [];
                        const pulses = [];

                        // Create nodes (neurons)
                        layers.forEach((layer, layerIndex) => {
                            const neurons = [];
                            for (let i = 0; i < layer.neurons; i++) {
                                const y = (i - (layer.neurons - 1) / 2) * 50;
                                const geometry = new THREE.SphereGeometry(5, 32, 32);
                                const material = new THREE.MeshBasicMaterial({ color: 0x00ffcc });
                                const sphere = new THREE.Mesh(geometry, material);
                                sphere.position.set(layer.x, y, 0);
                                scene.add(sphere);
                                neurons.push(sphere);
                            }
                            nodes.push(neurons);
                        });

                        // Create edges (connections) between layers
                        for (let l = 0; l < layers.length - 1; l++) {
                            for (let i = 0; i < layers[l].neurons; i++) {
                                for (let j = 0; j < layers[l + 1].neurons; j++) {
                                    const start = nodes[l][i].position;
                                    const end = nodes[l + 1][j].position;
                                    const geometry = new THREE.BufferGeometry().setFromPoints([
                                        start, end
                                    ]);
                                    const material = new THREE.LineBasicMaterial({ color: 0x00e6e6, transparent: true, opacity: 0.3 });
                                    const line = new THREE.Line(geometry, material);
                                    scene.add(line);
                                    edges.push({ line, start, end });
                                }
                            }
                        }

                        // Create pulses (traveling signals)
                        function createPulse(start, end) {
                            const geometry = new THREE.SphereGeometry(3, 16, 16);
                            const material = new THREE.MeshBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.8 });
                            const pulse = new THREE.Mesh(geometry, material);
                            pulse.position.copy(start);
                            scene.add(pulse);
                            return { pulse, start, end, t: 0 };
                        }

                        // Camera position
                        camera.position.z = 300;

                        // Animation loop
                        function animate() {
                            requestAnimationFrame(animate);

                            // Animate pulses
                            if (Math.random() < 0.05) { // Randomly spawn pulses
                                const edge = edges[Math.floor(Math.random() * edges.length)];
                                pulses.push(createPulse(edge.start, edge.end));
                            }

                            pulses.forEach((pulse, index) => {
                                pulse.t += 0.02;
                                if (pulse.t > 1) {
                                    scene.remove(pulse.pulse);
                                    pulses.splice(index, 1);
                                    return;
                                }
                                pulse.pulse.position.lerpVectors(pulse.start, pulse.end, pulse.t);
                                pulse.pulse.material.opacity = 0.8 * (1 - pulse.t);
                            });

                            // Rotate the network for a dynamic effect
                            scene.rotation.y += 0.005;

                            renderer.render(scene, camera);
                        }
                        animate();

                        // Handle window resize
                        window.addEventListener('resize', () => {
                            renderer.setSize(container.clientWidth, 300);
                            camera.aspect = container.clientWidth / 300;
                            camera.updateProjectionMatrix();
                        });
                    } catch (error) {
                        console.error("Error in start3DNeuralNetwork:", error);
                        container.innerHTML = '<div id="visualization-error">Failed to initialize 3D visualization. Check console for details.</div>';
                    }
                });
            }

            // Matrix Rain Background
            function createMatrixRain() {
                try {
                    console.log("Starting Matrix Rain...");
                    const canvas = document.createElement('canvas');
                    canvas.id = 'matrix-bg';
                    document.body.appendChild(canvas);
                    const ctx = canvas.getContext('2d');
                    if (!ctx) {
                        console.error("Canvas 2D context not supported.");
                        return;
                    }
                    canvas.height = window.innerHeight;
                    canvas.width = window.innerWidth;
                    const chars = '0101';
                    const fontSize = 14;
                    const columns = canvas.width / fontSize;
                    const drops = [];
                    for (let x = 0; x < columns; x++) {
                        drops[x] = 1;
                    }
                    function draw() {
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = '#00ffcc';
                        ctx.font = fontSize + 'px monospace';
                        for (let i = 0; i < drops.length; i++) {
                            const text = chars.charAt(Math.floor(Math.random() * chars.length));
                            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975)
                                drops[i] = 0;
                            drops[i]++;
                        }
                    }
                    setInterval(draw, 33);
                } catch (error) {
                    console.error("Error in createMatrixRain:", error);
                }
            }

            // Show loader during generation
            function showLoader() {
                try {
                    const loader = document.getElementById('loader');
                    if (loader) {
                        loader.style.display = 'block';
                    }
                } catch (error) {
                    console.error("Error in showLoader:", error);
                }
            }
            function hideLoader() {
                try {
                    const loader = document.getElementById('loader');
                    if (loader) {
                        loader.style.display = 'none';
                    }
                } catch (error) {
                    console.error("Error in hideLoader:", error);
                }
            }

            // Attach event listeners to the generate button
            function setupButtonListeners() {
                waitForElement('generate-button', () => {
                    try {
                        console.log("Setting up button listeners...");
                        const buttons = document.getElementsByTagName('button');
                        let generateButton = null;
                        for (let btn of buttons) {
                            if (btn.textContent === 'Initialize Identity Generation') {
                                generateButton = btn;
                                break;
                            }
                        }
                        if (generateButton) {
                            generateButton.id = 'generate-button'; // Ensure the button has an ID
                            generateButton.addEventListener('click', () => {
                                showLoader();
                            });

                            const outputDiv = document.querySelector('.dataframe');
                            if (outputDiv) {
                                const observer = new MutationObserver((mutations) => {
                                    mutations.forEach(() => {
                                        hideLoader();
                                    });
                                });
                                observer.observe(outputDiv, { childList: true, subtree: true });
                            }
                        }
                    } catch (error) {
                        console.error("Error in setupButtonListeners:", error);
                    }
                });
            }

            // Initialize everything
            function initializeVisualizations() {
                console.log("Initializing visualizations...");
                startNeuralAnimation();
                start3DNeuralNetwork();
                createMatrixRain();
                setupButtonListeners();
            }

            // Start everything
            setTimeout(() => {
                console.log("Starting initialization...");
                initializeVisualizations();
            }, 3000);
        });
    </script>
    """)
    gr.HTML('<div id="loader"></div>')
    gr.Markdown("# Neural Identity Matrix")
    gr.Markdown("Generate futuristic clone identities with an evolving AI core.")
    num_identities = gr.Slider(minimum=1, maximum=10, value=10, step=1, label="Number of Identities to Generate")
    resume_training = gr.Checkbox(label="Resume Training from Checkpoint", value=False)
    generate_button = gr.Button("Initialize Identity Generation")
    progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progress", interactive=False)
    status_message = gr.Markdown("Ready to Generate")
    output = gr.Dataframe(label="Identity Matrix Output")
    download_button = gr.File(label="Download Identities as CSV", visible=False)
    gr.HTML('<div id="neural-activity"></div>')
    gr.HTML('<div id="neural-network-3d"></div>')
    gr.Markdown("**Core Status**: Active and Evolving")

    generate_button.click(
        fn=generate_identities_gui,
        inputs=[num_identities, resume_training],
        outputs=[output, download_button, progress_bar, status_message]
    )

# Launch the app
demo.launch(share=False)