# Neural Identity Matrix (NIM)

**Create futuristic clone identities with AI-generated profiles and images.** The Neural Identity Matrix (NIM) is a Python-based tool that goes beyond standard text-to-image generation. It crafts detailed personas for futuristic clones, including names, ages, professions, cosmic traits (e.g., “Nebula Kitten” pet), and AI-crafted images using ComfyUI and the FLUX.1 model(s). Users can customize clone designs dynamically via a Gradio GUI, making it a powerful text-to-image and prompt-to-story generator.

## Features

- **Identity Generation**: Generate up to 250 clone profiles with attributes like name, age, profession, height, weight, cosmic traits (e.g., aura, tattoo, pet), and more.
- **Image Generation**: Create AI-crafted clone images using ComfyUI and FLUX.1 [dev], customizable with style themes (e.g., “Galactic Royalty”), locations (e.g., “Cosmic Nebula”), and overall themes (e.g., “Ethereal Dreamscape”).
- **Gradio Interface**: User-friendly GUI with dropdowns for styles, locations, and themes, a table for viewing identities, an image gallery, and a training loss plot.
- **Social Sharing**: Share generated images to X with auto-suggested captions.
- **Open-Source Datasets**: Includes `style_themes.csv`, `locations.csv`, `overall_themes.csv`, and more for customizable clone traits.

## Getting Started

### Prerequisites
- **Python**: 3.10.6 or higher
- **Conda**: For environment management
- **ComfyUI**: Running locally at `http://127.0.0.1:8188` for image generation
- **X API Credentials**: For sharing to X (optional)
- **Hardware**: GPU recommended for image generation with Flux-Dev.

## Getting Started
1. You need Python 3.10.6
2. Clone this repository: `git clone https://github.com/CRCODE22/Neural-Identity-Matrix.git`
3. Install dependencies: `pip install -r requirements.txt`
4. You need https://github.com/rgthree/rgthree-comfy in \ComfyUI\custom_nodes ![image](https://github.com/user-attachments/assets/54b5a7e1-e62c-4970-98bb-f01035b6c7c7)
6. Set up your environment (see `requirements.txt` for details) and ensure ComfyUI is running locally at `http://127.0.0.1:8188`.
7. Place the required `.csv` files (`dataset.csv`, `previous_names.csv`, `upper_clothing.csv`, `lower_clothing.csv`, `footwear.csv`, `style_themes.csv`, `locations.csv`, `overall_themes.csv`) in the project directory.
8. Verify that you have the necessary ComfyUI Custom Nodes, and that you can generate images. open CLN-010_SparPro_00001_.png as the workflow in ComfyUI and try to generate an image.
9. Run the script: `python Neural_Identity_Matrix.py` or `python app.py`
10. Open the Gradio interface in your browser and start generating clones!

## Flux Requirements:

Models used in Neural_Identity_Matrix.py

flux1-dev-fp8-e4m3fn.safetensors or acornIsSpinningFLUX_devfp8V11.safetensors or any other fp8-e4m3fn flux model.
https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e4m3fn.safetensors
https://huggingface.co/datasets/John6666/flux1-backup-202408/blob/main/acornIsSpinningFLUX_devfp8V11.safetensors

t5xxl_fp16.safetensors
https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

clip_l.safetensors
https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors

clip_g.safetensors
https://huggingface.co/second-state/stable-diffusion-3.5-large-GGUF/blob/main/clip_g.safetensors

You can use other Flux Models by modifying Neural_Identity_Matrix.py.
Search for: "inputs": {"unet_name": "flux1-dev-fp8-e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"},

![image](https://github.com/user-attachments/assets/185a256f-bab5-4c06-a271-789f1a93b9a7)

![image](https://github.com/user-attachments/assets/e8ee598e-6968-4870-ba7a-ec81af9b7ad4)

![image](https://github.com/user-attachments/assets/b37ddfe9-90b5-4bbd-84b1-e666ced0dcd9)

![image](https://github.com/user-attachments/assets/03488f1c-5e05-4cf5-88b2-3a36db32f6e2)

![image](https://github.com/user-attachments/assets/3ce302ef-82c6-44b0-bdcf-50f1d11c0d0f)

![image](https://github.com/user-attachments/assets/f0cd18e9-d81c-4e8b-9d0f-92227e214149) ![image](https://github.com/user-attachments/assets/e056968d-9f70-4e47-80d2-2931d6fa715b) ![image](https://github.com/user-attachments/assets/35c7dc63-7f3b-48eb-b87d-13f0cc827bc2)

## Acknowledgments
- Created by CRCODE22 and Grok 3 (xAI).
- Powered by PyTorch, Gradio, and ComfyUI for a seamless AI experience.
