# Import necessary libraries
from transformers import BlipProcessor, BlipForConditionalGeneration  # For image captioning
from PIL import Image  # To handle image processing
import gradio as gr  # Gradio for creating web interface
from diffusers import StableDiffusionPipeline  # For generating images from text
import torch  # To enable GPU acceleration
from deep_translator import GoogleTranslator  # For automatic translation
from gtts import gTTS  # Google Text-to-Speech for generating audio from text
import os  # For file handling

# Function to translate text to a target language
def translate_text(text, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(text)

# Load the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the Stable Diffusion pipeline for text-to-image generation
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)

# Check if GPU is available and move models to GPU for faster computation
if torch.cuda.is_available():
    model.to("cuda")
    pipe.to("cuda")

# Function for generating captions from images, with text-to-speech
def generate_caption(image, target_language):
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    translated_caption = translate_text(caption, target_language)
    
    tts = gTTS(text=translated_caption, lang=target_language)
    tts.save("output.mp3") 
    
    return translated_caption, "output.mp3"

# Function for generating images from text descriptions
def generate_image_from_text(description, image_input=None):
    translated_description = translate_text(description, "en")

    if image_input is not None:
        inputs = processor(image_input, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs)
        image_caption = processor.decode(out[0], skip_special_tokens=True)
        translated_description += " " + image_caption  # Append the image caption to the text

    image = pipe(translated_description, num_inference_steps=70, guidance_scale=6.5).images[0]
    
    return image

# Gradio interface for image captioning with Text-to-Speech
iface_image_to_text = gr.Interface(
    fn=generate_caption,  # The function to execute
    inputs=[
        gr.Image(type="pil"),  # Image input
        gr.Dropdown(
            choices=[("English", "en"), ("Arabic", "ar"), ("French", "fr"), ("Spanish", "es"), ("German", "de")],
            label="Select Output Language",
            value="ar"  # Default language is set to Arabic
        )
    ],
    outputs=[
        "text",  # Text output (caption)
        gr.Audio(type="filepath")  # Audio output (speech of caption)
    ],
    title="Image Captioning with Speech Output",
    description="Upload an image and get a caption in the selected language with audio output.",
    
    # Adding examples (local images paths or URLs)
    examples=[
        ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoQVWVsG_u0dE0IDFQTszJRcSz1kl3PlXb_g&s", "ar"],
        ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwf8ck2f453p_Nt3jJjo6Xfl5Iu4IpprnLJw&s", "en"]
    ]
)


# Example inputs for text-to-image generation
examples = [
    ["A beautiful sunset over the mountains."],
    ["A futuristic cityscape with flying cars."],
    ["منظر للبحر مع قارب صغير يطفو على سطح الماء"],
    ["رجل يقرأ كتابًا تحت شجرة في يوم مشمس"],
    ["A spaceship landing on Mars."],
    ["سفينة تبحر في محيط هادئ تحت سماء زرقاء صافية"],
]

# Gradio interface for text-to-image generation with examples
iface_text_to_image = gr.Interface(
    fn=generate_image_from_text,
    inputs=[
        gr.Textbox(label="Enter your description", placeholder="Type your description here..."),
        gr.Image(type="pil", label="Optional: Upload an image")
    ],
    outputs="image",
    title="Text-to-Image Generation",
    description="Enter a description or upload an image to generate an image.",
    examples=examples  # Added examples for quick access
)

# Combining both interfaces into a tabbed Gradio interface
iface_combined = gr.TabbedInterface(
    [iface_image_to_text, iface_text_to_image],
    tab_names=["Image Captioning with Speech", "Text-to-Image"]
)

# Launch the Gradio interface
iface_combined.launch()
