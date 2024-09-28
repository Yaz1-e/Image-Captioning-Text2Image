# Image-Captioning-Text2Image
This project allows users to either upload an image and receive an auto-generated caption with speech output (Image Captioning), or input a text description to generate an image (Text-to-Image). The project leverages advanced machine learning models such as BLIP for image captioning and Stable Diffusion for generating images from text.

## Models Used
- **BLIP** (Bootstrapping Language-Image Pretraining) for image captioning: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **Stable Diffusion 2.1** for text-to-image generation: [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- **Deep-Translator** for automatic translation: [deep-translator](https://pypi.org/project/deep-translator/)
- **gTTS** (Google Text-to-Speech) for converting captions into speech: [gTTS](https://pypi.org/project/gTTS/)

## Key Features
1. **Image Captioning with Speech Output**: Automatically generates a description of an uploaded image, translates it into a selected language, and converts the caption into speech.
2. **Text-to-Image Generation**: Users can either provide a text description or choose from pre-made suggestions to generate a new image.
3. **Multi-language Support**: Caption translation supports multiple languages (English, Arabic, French, Spanish, German).
4. **GPU Support**: If available, the system utilizes GPU resources to accelerate the model's processing time.

## Requirements
You need to have the following libraries installed:
- `transformers` (for BLIP and Stable Diffusion models)
- `diffusers` (for text-to-image generation)
- `torch` (for handling model inference and processing)
- `PIL` (for image processing)
- `gradio` (for building the user interface)
- `deep-translator` (for automatic caption translation)
- `gtts` (for converting translated text to speech)

You can install them by running the following command:
```bash
pip install transformers diffusers torch pillow gradio deep-translator gtts
```

## How to Run the Project
You can use the link below for Hugging Face or you can download the script.


## Usage
**Image Captioning**
Upload an image to get an auto-generated caption and its speech output.

**Text-to-Image**
Enter a text description or choose a suggestion from the dropdown to generate an image.

## Example

**Image Captioning:**
- **Input:** Upload an image of a sunset.
- **Output:**
  - **Text:** "A beautiful sunset over the mountains."
  - **Audio:** Speech output of the caption.

**Text-to-Image:**
- **Input:** Text description: "A robot playing chess with a human."
- **Output:** An image of a robot playing chess with a human.




## Known Issues
Without GPU support, the processing time may significantly increase.

Some languages may not be fully supported for speech output.

## Performance Tips
If there's a slowdown in translation or image generation, you can use cloud computing services like Google Colab for better performance.

If you have a GPU, ensure that the models are utilizing it to speed up processing.


## Tutorial Video
A tutorial video demonstrating the project is available in the downloads folder. You can find it as ****** for a detailed walkthrough on how to use this Image-Captioning-Text2Image interface.



## Demo

You can try out the project on Hugging Face Spaces: [Demo on Hugging Face](https://huggingface.co/spaces/Yaz1-e/Image-Captioning-Text2Image)

