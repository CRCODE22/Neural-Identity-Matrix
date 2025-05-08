# --- Start of generate_dataset.py (Updated for Neural Identity Matrix V24.22) ---
# Run `python -m py_compile generate_dataset.py` to check syntax before execution
# This script generates a synthetic dataset (dataset.csv) for Neural_Identity_Matrix_V24.22.py
# Saves dataset.csv in the current directory (e.g., K:\My Github Repositories\Neural-Identity-Matrix\)
# Requires: pandas, numpy

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
num_samples = 100

# Updated professions list to match V24.21
professions = [
    'Architect', 'Athlete', 'Blogger', 'Cosplayer', 'Dancer', 'Entrepreneur',
    'Fashion Designer', 'Fitness Coach', 'Glamour Model', 'Graphic Designer',
    'Illustrator', 'Influencer', 'Interior Designer', 'Inventor', 'Journalist',
    'Makeup Artist', 'Photographer', 'Poet', 'Producer', 'Programmer', 'Singer',
    'Social Media Star', 'Tattoo Artist', 'Youtuber',
    'Astrologer', 'Chef', 'DJ', 'Engineer', 'Gamer', 'Hacker', 'Pilot',
    'Scientist', 'Streamer', 'Writer', 'Camgirl', 'Actress', 'Cop', 'High Priestess' 'Witch'
]
weights = [1/len(professions)] * len(professions)

# Expanded name lists for diversity, added names with "X"
first_names_pool = [
    'Mila', 'Erica', 'Sophie', 'Clara', 'Phi', 'Gemma', 'Luna', 'Zoe',
    'Aisha', 'Hana', 'Isabella', 'Mei', 'Olivia', 'Fatima', 'Nancy', 'Mia',
    'Lena', 'Sofia', 'Emma', 'Jeanette', 'Katherine', 'Andie', 'Ali', 'Nicoletta',
    'Marion', 'Lydia', 'Erin', 'Janet', 'Shirley', 'Yuki', 'Amara', 'Elena',
    'Priya', 'Nia', 'Sana', 'Tara', 'Leila', 'Rin', 'Ava', 'Chloe',
    'Xena', 'Ximena'  # Added names starting with "X"
]

last_names_pool = [
    'Azul', 'Campbell', 'Kovalenko', 'Reject', 'Four', 'Three', 'Spark', 'Nova',
    'MacDonald', 'MacDowell', 'MacGraw', 'Machiavelli', 'Mack', 'Mackay', 'MacLaine',
    'Patel', 'Sato', 'Rodriguez', 'Chen', 'James', 'Al-Sayed', 'Van Dijk', 'Ace',
    'Malkova', 'Smith', 'Garcia', 'Kim', 'Singh', 'Nguyen', 'Lopez', 'Khan',
    'Moreno', 'Tanaka', 'Ali', 'Park', 'Cruz', 'Reyes', 'Roovers', 'Hamers', 'Gupta', 'Yamamoto',
    'Xenakis'  # Added last name starting with "X"
]

# Base nicknames (min length 3, max length 10 for base)
base_nicknames_pool = [
    'Mil', 'Eri', 'Soph', 'Cla', 'Phi', 'Gem', 'Lun', 'Zoe',
    'Ais', 'Han', 'Isa', 'Mei', 'Oli', 'Fat', 'Nan', 'Mia',
    'Len', 'Sof', 'Emm', 'Jea', 'Kat', 'And', 'Nic', 'Mar',
    'Lyd', 'Jan', 'Shi', 'Yuki', 'Ama', 'Elen', 'Pri', 'Nia',
    'San', 'Tar', 'Lei', 'Rin', 'Ava', 'Chlo',
    'Xen'  # Added nickname starting with "X"
]

# Nickname suffixes (same as V24.21)
nickname_suffixes = [
    'Star', 'Cosmo', 'Dreamer', 'Vibe', 'Guru', 'Nebula', 'Quantum', 'Spark', '42',
    'Player', 'GamerX', 'Pro',
    'ModelX', 'Starlet', 'Glam',
    'Clone', 'NIM', 'Core'
]

# Generate synthetic data
data = {
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
    'Nationality': np.random.choice(
        [
            'American', 'British', 'French', 'Japanese', 'Australian', 'Dutch',
            'Russian', 'Ukrainian', 'Indian', 'Brazilian', 'Chinese', 'Mexican',
            'Korean', 'Egyptian', 'Canadian', 'German', 'Italian', 'Spanish'
        ],
        num_samples
    ).tolist(),
    'Ethnicity': np.random.choice(
        ['Caucasian', 'Asian', 'African', 'Mixed', 'Latina', 'South Asian', 'Middle Eastern'],
        num_samples
    ).tolist(),
    'Birthplace': np.random.choice(
        [
            'New York, USA', 'London, UK', 'Paris, France', 'Tokyo, Japan',
            'Amsterdam, Netherlands', 'Moscow, Russia', 'Shanghai, China',
            'Sao Paulo, Brazil', 'Mumbai, India', 'Sydney, Australia',
            'Mexico City, Mexico', 'Seoul, South Korea', 'Cairo, Egypt',
            'Toronto, Canada', 'Berlin, Germany', 'Rome, Italy', 'Madrid, Spain'
        ],
        num_samples
    ).tolist(),
    'Profession': np.random.choice(professions, num_samples, p=weights).tolist(),
    'Height': np.random.uniform(150, 180, num_samples).tolist(),
    'Weight': np.random.uniform(45, 70, num_samples).tolist(),
    'Body type': np.random.choice(['Slim', 'Athletic', 'Curvy', 'Petite'], num_samples).tolist(),
    'Body Measurements': [
        f"{np.random.randint(80, 100)}-{np.random.randint(55, 65)}-{np.random.randint(85, 105)}"
        for _ in range(num_samples)
    ],
    'Hair color': np.random.choice(
        ['Blonde', 'Brown', 'Black', 'Red', 'Auburn', 'Purple'],
        num_samples
    ).tolist(),
    'Eye color': np.random.choice(
        ['Blue', 'Green', 'Brown', 'Hazel', 'Amber', 'Violet'],
        num_samples
    ).tolist(),
    'Bra/cup size': np.random.choice(['A', 'B', 'C', 'D', 'DD'], num_samples).tolist(),
    'Boobs': np.random.choice(['Natural', 'Enhanced'], num_samples).tolist(),
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure numerical columns are properly formatted
numerical_columns = ['Age', 'Height', 'Weight']
for col in numerical_columns:
    df[col] = df[col].astype(float).round(0).astype(int)

# Debug: Print column types, value ranges, and sample nicknames
print("Dataset Summary:")
for col in df.columns:
    if col in numerical_columns:
        print(f"{col}: Type={df[col].dtype}, Range=({df[col].min()}, {df[col].max()})")
    else:
        print(f"{col}: Type={df[col].dtype}, Unique Values={len(df[col].unique())}")
print(f"Sample Nicknames (first 5): {df['Nickname'].head().tolist()}")

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"Generated dataset.csv with {len(df)} rows")

# --- End of generate_dataset.py ---