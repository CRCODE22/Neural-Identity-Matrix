# Neural Identity Matrix

Welcome to the **Neural Identity Matrix** project! This is a futuristic AI-powered identity generator that creates detailed synthetic identities for fictional "clones." Built using Python, PyTorch, and Gradio, this project leverages neural networks to generate creative and realistic identity profiles, including names, nicknames, and other personal attributes. The project is designed to evolve over time, with plans to integrate advanced features like AI-generated ID images, Selfies, Photos using Stable Diffusion and Videos.

The Neural Identity Matrix is perfect for creative projects, simulations, or anyone interested in exploring the capabilities of neural networks in generating synthetic data with a futuristic twist.

## Table of Contents
- [Project Overview](#project-overview)
- [Current Features](#current-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Neural Identity Matrix is a machine learning project that generates synthetic identities for futuristic "clones." It uses a combination of neural networks to create detailed profiles, including:

- **Personal Details**: First name, last name, nickname, age, birth date, nationality, ethnicity, and birthplace.
- **Professional Details**: Profession (e.g., Doctor, Artist, Engineer).
- **Physical Characteristics**: Height, weight, body type, body measurements, hair color, eye color, bra/cup size, and whether the boobs are natural or enhanced.

The core of the project consists of two neural networks:
1. **IdentityGenerator**: A feedforward neural network that generates numerical and categorical attributes (e.g., age, height, nationality).
2. **NameGeneratorRNN**: A character-level Recurrent Neural Network (RNN) using GRU layers to generate creative first names, last names, and nicknames by learning patterns from existing names.

The project includes a user-friendly Gradio interface with a Matrix-inspired theme, featuring a progress bar, training loss visualization, and a downloadable CSV of generated identities.

## Current Features

- **Synthetic Identity Generation**:
  - Generates detailed profiles for up to 250 identities at a time.
  - Includes fields like `Firstname`, `Lastname`, `Nickname`, `Age`, `Born`, `Nationality`, `Ethnicity`, `Birthplace`, `Profession`, `Height`, `Weight`, `Body type`, `Body Measurements`, `Hair color`, `Eye color`, `Bra/cup size`, and `Boobs`.
  - Identities are filtered to ensure realistic values (e.g., age between 18 and 40, height between 150 and 180 cm).

- **Creative Name and Nickname Generation**:
  - Uses a character-level RNN to generate new first names, last names, and nicknames that are not limited to a predefined list.
  - Ensures generated names are at least 3 characters long and contain at least one vowel for plausibility.
  - Avoids duplicate last names within a single generation run for diversity.

- **Training and Early Stopping**:
  - Trains the `IdentityGenerator` neural network over multiple cycles (up to 10 cycles, 50 epochs each).
  - Implements early stopping to prevent overfitting, with a patience of 20 epochs.
  - Displays training progress with a loss plot and detailed console output (e.g., average loss, epoch time, learning rate).

- **Gradio Interface**:
  - A sleek, Matrix-themed UI with a progress bar, training loss visualization, and a table of generated identities.
  - Allows users to specify the number of identities to generate (1 to 250).
  - Supports resuming training from a checkpoint.
  - Outputs a downloadable CSV file (`generated_cha_identities.csv`) with the generated identities.

- **Checkpointing**:
  - Saves the best model during training (`best_model.pth`) based on the lowest loss.
  - Supports resuming training from a previous checkpoint (`checkpoint_model.pth`).

## Installation

### Prerequisites

To run the Neural Identity Matrix, you’ll need the following:

- **Python**: Version 3.8 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **Hardware**: A CPU is sufficient, but a GPU with CUDA support is recommended for faster training.
- **Dependencies**: The project relies on several Python libraries, which are listed in the `requirements.txt` file (see below).

#### Required Libraries

More will be added later below see some examples:

While it is training example:

![image](https://github.com/user-attachments/assets/45944c19-1bf5-4b49-aaec-20fba575311f) Updated 05/05/2025

2D and 3D Neural Network Animations are now working and Matrix Falling Character Rain.
Once you have Gradio running locally open index.html from a local folder on your computer and the Gradio interface will be integrated through iframe

![image](https://github.com/user-attachments/assets/60ab355c-6c83-47d3-b3eb-22821a675954) Updated 05/05/2025

You can resume by checking Resume Training from Checkpoint

![image](https://github.com/user-attachments/assets/9c3f5043-1abb-4fb2-8837-28873d95f819)

Here is a screenshot of the data from the created: checkpoint_learned_identities.csv

![image](https://github.com/user-attachments/assets/8423b523-afb5-4b93-ae78-01e507091a04)

Here is a screenshot of generated identities.

![image](https://github.com/user-attachments/assets/34e90941-d98c-447c-8dd5-027a7c013b1d) Updated 05/05/2025

Here is a screenshot of the Anaconda console while training and completed:

![image](https://github.com/user-attachments/assets/f082fa63-8492-416e-bca9-58bb48c2f619)
