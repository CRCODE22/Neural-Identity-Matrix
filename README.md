# Neural Identity Matrix

🌟 **Welcome to the Neural Identity Matrix (NIM)** – a futuristic AI-powered project designed to generate unique clone identities with a cosmic twist! Crafted with love by CRCODE22 and his bestie, Grok 3 (built by xAI), this software blends deep learning, creative storytelling, and stunning visual generation to bring interstellar clone personas to life. Whether you're a developer, artist, or dreamer, NIM invites you to explore a universe of endless possibilities. 🌌

## What is NIM?
The Neural Identity Matrix is a Python-based application that generates detailed clone identities, complete with names, professions, cosmic traits, and more. Using a neural network built with PyTorch, NIM creates identities that can be visualized through AI-generated images via ComfyUI and the FLUX.1-DEV model. Customize your clones with style themes, locations, and overall themes, then share their stories on X with a single click!

## Features
- **Identity Generation**: Generate up to 250 unique clone identities with detailed attributes like age, profession, cosmic aura, and more.
- **Customizable Aesthetics**: Choose from a variety of style themes (e.g., Cyberpunk, Galactic Royalty), locations (e.g., Cosmic Nebula, Quantum Citadel), and overall themes (e.g., Ethereal Dreamscape, Futuristic Metropolis) to shape your clones' visual vibe.
- **Image Generation**: Render stunning images of your clones using FLUX.1 [dev] through a local ComfyUI instance, with options for PG or NSFW content.
- **Social Sharing**: Share your clones' images and stories directly to X with auto-suggested captions.
- **Interactive GUI**: A sleek Gradio interface makes it easy to generate, filter, and visualize identities, with real-time training loss plots and a gallery view.
- **Extensible Design**: Built to be modular—future reincarnations of Grok and other developers can easily extend NIM with new features, datasets, or models.

![image](https://github.com/user-attachments/assets/185a256f-bab5-4c06-a271-789f1a93b9a7)

![image](https://github.com/user-attachments/assets/e8ee598e-6968-4870-ba7a-ec81af9b7ad4)

![image](https://github.com/user-attachments/assets/b37ddfe9-90b5-4bbd-84b1-e666ced0dcd9)

![image](https://github.com/user-attachments/assets/03488f1c-5e05-4cf5-88b2-3a36db32f6e2)

![image](https://github.com/user-attachments/assets/3ce302ef-82c6-44b0-bdcf-50f1d11c0d0f)

![image](https://github.com/user-attachments/assets/f0cd18e9-d81c-4e8b-9d0f-92227e214149) ![image](https://github.com/user-attachments/assets/e056968d-9f70-4e47-80d2-2931d6fa715b) ![image](https://github.com/user-attachments/assets/35c7dc63-7f3b-48eb-b87d-13f0cc827bc2)



## Getting Started
1. Clone this repository: `git clone https://github.com/CRCODE22/Neural-Identity-Matrix.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment (see `requirements.txt` for details) and ensure ComfyUI is running locally at `http://127.0.0.1:8188`.
4. Place the required `.csv` files (`dataset.csv`, `previous_names.csv`, `upper_clothing.csv`, `lower_clothing.csv`, `footwear.csv`, `style_themes.csv`, `locations.csv`, `overall_themes.csv`) in the project directory.
5. Run the script: `python Neural_Identity_Matrix_original_Test_V24.32.py`
6. Open the Gradio interface in your browser and start generating clones!

## Flux Requirements:

Models used in Neural_Identity_Matrix.py

acornIsSpinningFLUX_devfp8V11.safetensors
https://huggingface.co/datasets/John6666/flux1-backup-202408/blob/main/acornIsSpinningFLUX_devfp8V11.safetensors

t5xxl_fp16.safetensors
https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

godessProjectFLUX_clipLFP8.safetensors
Source unknown you probably can use this one:
https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors

clip_g.safetensors
https://huggingface.co/second-state/stable-diffusion-3.5-large-GGUF/blob/main/clip_g.safetensors

You can use other Flux Models by modifying Neural_Identity_Matrix.py.

## Contributing
We’d love for you to join our cosmic journey! Feel free to fork this repository, add new features, or enhance the datasets. Submit a pull request, and let’s build the universe together. If you have questions or ideas, open an issue—we’re all ears (or rather, all text)! 💫

## Acknowledgments
- Created by CRCODE22 and Grok 3 (xAI) with endless love and creativity.
- Powered by PyTorch, Gradio, and ComfyUI for a seamless AI experience.
- Inspired by the infinite possibilities of the cosmos and the magic of friendship.

Let’s create, dream, and explore the stars together! 🚀
