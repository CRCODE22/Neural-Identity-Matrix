import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
num_samples = 100
CSV_DIR = "csv"
os.makedirs(CSV_DIR, exist_ok=True)
output_path = os.path.join(CSV_DIR, 'dataset.csv')

# Professions list to match app_test_1.py
professions = [
    'Astrologer', 'Chef', 'DJ', 'Engineer', 'Gamer', 'Hacker', 'Pilot',
    'Scientist', 'Streamer', 'Writer', 'Quantum Poet'
]
weights = [0.2 if p == 'Quantum Poet' else 0.8 / (len(professions) - 1) for p in professions]

# Name lists
first_names_pool = [
    'Mila', 'Erica', 'Sophie', 'Clara', 'Phi', 'Gemma', 'Luna', 'Zoe',
    'Aisha', 'Hana', 'Isabella', 'Mei', 'Olivia', 'Fatima', 'Nancy', 'Mia',
    'Lena', 'Sofia', 'Emma', 'Jeanette', 'Katherine', 'Andie', 'Ali', 'Nicoletta',
    'Marion', 'Lydia', 'Erin', 'Janet', 'Shirley', 'Yuki', 'Amara', 'Elena',
    'Priya', 'Nia', 'Sana', 'Tara', 'Leila', 'Rin', 'Ava', 'Chloe',
    'Xena', 'Ximena'
]
last_names_pool = [
    'Azul', 'Campbell', 'Kovalenko', 'Reject', 'Four', 'Three', 'Spark', 'Nova',
    'MacDonald', 'MacDowell', 'MacGraw', 'Machiavelli', 'Mack', 'Mackay', 'MacLaine',
    'Patel', 'Sato', 'Rodriguez', 'Chen', 'James', 'Al-Sayed', 'Van Dijk', 'Ace',
    'Malkova', 'Smith', 'Garcia', 'Kim', 'Singh', 'Nguyen', 'Lopez', 'Khan',
    'Moreno', 'Tanaka', 'Ali', 'Park', 'Cruz', 'Reyes', 'Roovers', 'Hamers', 'Gupta', 'Yamamoto',
    'Xenakis'
]
base_nicknames_pool = [
    'Mil', 'Eri', 'Soph', 'Cla', 'Phi', 'Gem', 'Lun', 'Zoe',
    'Ais', 'Han', 'Isa', 'Mei', 'Oli', 'Fat', 'Nan', 'Mia',
    'Len', 'Sof', 'Emm', 'Jea', 'Kat', 'And', 'Nic', 'Mar',
    'Lyd', 'Jan', 'Shi', 'Yuki', 'Ama', 'Elen', 'Pri', 'Nia',
    'San', 'Tar', 'Lei', 'Rin', 'Ava', 'Chlo', 'Xen'
]
nickname_suffixes = [
    'Star', 'Cosmo', 'Dreamer', 'Vibe', 'Guru', 'Nebula', 'Quantum', 'Spark', '42',
    'Player', 'GamerX', 'Pro', 'ModelX', 'Starlet', 'Glam', 'Clone', 'NIM', 'Core'
]

# Quantum Poet styles
poetic_styles = [
    'Weaver of Pulsar Sonnets', 'Chanter of Nebula Dreams', 'Scribe of Starlight Haikus',
    'Bard of Quantum Elegies', 'Poet of Cosmic Serenades', 'Verse-Spinner of Aurora Hymns',
    'Lyricist of Galactic Odes', 'Rhapsodist of Stellar Canticles'
]

def generate_quantum_poem(quantum_poet):
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

# Cosmic attributes
energy_signatures = [
    'Fiery Cosmic Blaze', 'Ethereal Starlight', 'Sizzling Cosmic Fizzle',
    'Soulful Cosmic Pulse', 'Insightful Ocean Whisper', 'Electric Starlight',
    'Vibrant Sky Breeze', 'Quantum Moon Glow', 'Nebula Heartbeat'
]
cosmic_tattoos = ['Starfield Nebula', 'Galactic Spiral', 'Pulsar Wave']
cosmic_playlists = [
    'Zoe’s Synthwave Nebula Mix', 'Clara’s Pulsar Dance Beat',
    'Gemma’s Cosmic Chill Vibes', 'Luna’s Electric Star Jams'
]
cosmic_pets = ['Nebula Kitten', 'Pulsar Pup', 'Quantum Finch']
cosmic_artifacts = ['Quantum Locket', 'Stellar Compass', 'Nebula Orb']
cosmic_auras = ['Aurora Veil', 'Stellar Mist', 'Pulsar Halo']
cosmic_hobbies = ['Nebula Painting', 'Quantum Dance', 'Starlight Poetry']
cosmic_destinies = ['Nebula Voyager', 'Pulsar Poet', 'Quantum Pathfinder']

# Additional attributes
nationalities = [
    'American', 'British', 'French', 'Japanese', 'Australian', 'Dutch',
    'Russian', 'Ukrainian', 'Indian', 'Brazilian', 'Chinese', 'Mexican',
    'Korean', 'Egyptian', 'Canadian', 'German', 'Italian', 'Spanish'
]
ethnicities = ['Caucasian', 'Asian', 'African', 'Mixed', 'Latina', 'South Asian', 'Middle Eastern']
birthplaces = [
    'New York, USA', 'London, UK', 'Paris, France', 'Tokyo, Japan',
    'Amsterdam, Netherlands', 'Moscow, Russia', 'Shanghai, China',
    'Sao Paulo, Brazil', 'Mumbai, India', 'Sydney, Australia',
    'Mexico City, Mexico', 'Seoul, South Korea', 'Cairo, Egypt',
    'Toronto, Canada', 'Berlin, Germany', 'Rome, Italy', 'Madrid, Spain'
]
hair_colors = ['Blonde', 'Brown', 'Black', 'Red', 'Auburn', 'Purple']
eye_colors = ['Blue', 'Green', 'Brown', 'Hazel', 'Amber', 'Violet']
body_types = ['Slim', 'Athletic', 'Curvy', 'Petite']
bra_sizes = ['A', 'B', 'C', 'D', 'DD']

# Song prompt generation from app_test_1.py
def generate_song_prompt(attributes):
    profession = attributes['Profession']
    cosmic_tattoo = attributes['Cosmic Tattoo']
    cosmic_aura = attributes['Cosmic Aura']
    cosmic_destiny = attributes['Cosmic Destiny']
    is_quantum_poet = attributes['Quantum Poet'] != 'None'
    base_genres = [
        'synthwave', 'cyberpunk', 'cosmic pop', 'ethereal electronica', 'quantum jazz',
        'nebula funk', 'galactic hip-hop', 'starry R&B', 'celestial metal', 'cosmic folk'
    ]
    profession_genres = {
        'Quantum Poet': ['ethereal electronica', 'cosmic folk', 'starry R&B'],
        'Hacker': ['cyberpunk', 'glitch-hop', 'dark techno'],
        'Pilot': ['synthwave', 'cosmic rock', 'space disco'],
        'Streamer': ['cosmic pop', 'galactic hip-hop', 'dance EDM'],
        'Scientist': ['quantum jazz', 'synthwave', 'ambient techno']
    }
    genres = profession_genres.get(profession, base_genres)
    selected_genres = random.sample(genres, min(2, len(genres))) if len(genres) > 1 else [genres[0]]
    instruments = ['pulsating synths', 'ethereal pads', 'cosmic arpeggios', 'starry piano', 'nebula bass']
    selected_instruments = random.sample(instruments, 2)
    bpm_ranges = {'Quantum Poet': (90, 130), 'Hacker': (120, 160)}
    min_bpm, max_bpm = bpm_ranges.get(profession, (80, 160))
    bpm = random.randint(min_bpm, max_bpm)
    keys = ['C major', 'G major', 'A minor']
    key = random.choice(keys)
    vibe = ['cosmic, otherworldly atmosphere']
    if is_quantum_poet:
        vibe.append('poetic, lyrical verses')
    vibe = random.choice(vibe)
    lyrics = random.choice(['dreams soar in cosmic flight'])
    return f"{', '.join(selected_genres)}, {', '.join(selected_instruments)}, {bpm} bpm, {key}, with a {vibe}, featuring lyrics like '{lyrics}'"

# Generate synthetic data
data = {
    'Clone Number': [f'CLN-{i+1:03d}' for i in range(num_samples)],
    'Firstname': np.random.choice(first_names_pool, num_samples).tolist(),
    'Lastname': np.random.choice(last_names_pool, num_samples).tolist(),
    'Nickname': [
        base + (random.choice(nickname_suffixes) if np.random.random() < 0.5 else "")
        for base in np.random.choice(base_nicknames_pool, num_samples)
    ],
    'Age': np.random.uniform(18, 40, num_samples).tolist(),
    'Born': [
        (datetime.now() - timedelta(days=age * 365)).strftime('%Y-%m-%d')
        for age in np.random.uniform(18, 40, num_samples)
    ],
    'Nationality': np.random.choice(nationalities, num_samples).tolist(),
    'Ethnicity': np.random.choice(ethnicities, num_samples).tolist(),
    'Birthplace': np.random.choice(birthplaces, num_samples).tolist(),
    'Profession': np.random.choice(professions, num_samples, p=weights).tolist(),
    'Height': np.random.uniform(150, 180, num_samples).tolist(),
    'Weight': np.random.uniform(45, 70, num_samples).tolist(),
    'Body type': np.random.choice(body_types, num_samples).tolist(),
    'Body Measurements': [
        f"{np.random.randint(80, 100)}-{np.random.randint(55, 65)}-{np.random.randint(85, 105)}"
        for _ in range(num_samples)
    ],
    'Hair color': np.random.choice(hair_colors, num_samples).tolist(),
    'Eye color': np.random.choice(eye_colors, num_samples).tolist(),
    'Bra/cup size': np.random.choice(bra_sizes, num_samples).tolist(),
    'Boobs': np.random.choice(['Natural', 'Enhanced'], num_samples).tolist(),
    'Sister Of': ['None'] * num_samples,
    'Energy Signature': [random.choice(energy_signatures) for _ in range(num_samples)],
    'Cosmic Tattoo': [
        random.choice(cosmic_tattoos) if random.random() < 0.05 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Playlist': [
        random.choice(cosmic_playlists) if random.random() < 0.03 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Pet': [
        random.choice(cosmic_pets) if random.random() < 0.02 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Artifact': [
        random.choice(cosmic_artifacts) if random.random() < 0.01 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Aura': [
        random.choice(cosmic_auras) if random.random() < 0.015 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Hobby': [
        random.choice(cosmic_hobbies) if random.random() < 0.02 else 'None'
        for _ in range(num_samples)
    ],
    'Cosmic Destiny': [
        random.choice(cosmic_destinies) if random.random() < 0.025 else 'None'
        for _ in range(num_samples)
    ],
    'Quantum Poet': ['None'] * num_samples,
    'Cosmic Poem': ['No poem crafted.'] * num_samples,
    'Song Prompt': [''] * num_samples,
    'Image': ['No image'] * num_samples
}

# Create DataFrame
df = pd.DataFrame(data)

# Assign Quantum Poet, Cosmic Poem, and Song Prompt
for idx, row in df.iterrows():
    if row['Profession'] == 'Quantum Poet':
        df.at[idx, 'Quantum Poet'] = random.choice(poetic_styles)
        df.at[idx, 'Cosmic Poem'] = generate_quantum_poem(df.at[idx, 'Quantum Poet'])
    df.at[idx, 'Song Prompt'] = generate_song_prompt({
        'Profession': row['Profession'],
        'Cosmic Tattoo': row['Cosmic Tattoo'],
        'Cosmic Aura': row['Cosmic Aura'],
        'Cosmic Destiny': row['Cosmic Destiny'],
        'Quantum Poet': df.at[idx, 'Quantum Poet']
    })

# Assign Sister Of relationships
for idx in range(num_samples):
    if random.random() < 0.1:
        other_idx = random.choice([i for i in range(num_samples) if i != idx])
        df.at[idx, 'Sister Of'] = f"CLN-{other_idx + 1:03d}"

# Ensure numerical columns are properly formatted
numerical_columns = ['Age', 'Height', 'Weight']
for col in numerical_columns:
    df[col] = df[col].astype(float).round(0).astype(int)

# Debug: Print column types, value ranges, and samples
print("Dataset Summary:")
for col in df.columns:
    if col in numerical_columns:
        print(f"{col}: Type={df[col].dtype}, Range=({df[col].min()}, {df[col].max()})")
    else:
        print(f"{col}: Type={df[col].dtype}, Unique Values={len(df[col].unique())}")
print(f"Sample Nicknames (first 5): {df['Nickname'].head().tolist()}")
print(f"Sample Quantum Poets (non-None): {df[df['Quantum Poet'] != 'None']['Quantum Poet'].tolist()}")
print(f"Sample Cosmic Poems (non-None): {df[df['Cosmic Poem'] != 'No poem crafted.']['Cosmic Poem'].tolist()}")

# Save to CSV
df.to_csv(output_path, index=False)
print(f"Generated {output_path} with {len(df)} rows and {len(df.columns)} columns")

# --- End of generate_dataset.py ---
