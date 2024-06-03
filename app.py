import streamlit as st
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch


def load_model() -> StableDiffusionUpscalePipeline:
    """
    Loads the stable diffusion upscale model.

    Returns:
        The loaded StableDiffusionUpscalePipeline model.
    """
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
    return pipeline

def model_upscale_image(pipeline: StableDiffusionUpscalePipeline, image: Image.Image, prompt: str) -> Image.Image:
    """
    Upscales the given image using the given model.

    Args:
        pipeline (StableDiffusionUpscalePipeline): The model to use for upscaling.
        image (Image.Image): The image to upscale.

    Returns:
        Image.Image: The upscaled image.
    """
    upscaled_image = pipeline(prompt=prompt, image=image).images[0]

    return upscaled_image

def upscale_image(image: Image.Image) -> None:
    """
    Function to upscale the given image using the model.

    Args:
        image (Image.Image): The image to upscale.
    """
    container = st.container(border=True)
    col1, col2 = container.columns(2)

    col2.info("Upscaling image...")

    # Load the model
    pipeline = load_model()

    # Define prompt
    prompt = "A cat picture"
    
    # Upscale the image
    upscaled_image = model_upscale_image(pipeline, image, prompt)

    col2.info("Image upscaled successfully!")
    
    # Display upscaled image
    col1.image(image, caption='Upscaled Image', width=300)

def image_uploader() -> Image.Image:
    """
    Function to upload and display an image using Streamlit.

    Returns:
        Image.Image: The uploaded image as a PIL Image object, or None if no image is uploaded.
    """
    container = st.container(border=True)

    uploaded_image = container.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    col1, col2 = container.columns(2)
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1.image(image, caption="Uploaded Image", width=300)
        
        # Display image dimensions and size
        image_size = image.size
        image_size_text = f"Image Dimensions: {image_size[0]}x{image_size[1]}"
        image_size_text += f"\n\nImage Size: {uploaded_image.size} bytes"
        
        col2.info(image_size_text)
        
        return image
    return None

def main():
    st.title("🖼️ Welcome to Image UpScaler!")
    # st.write("""
    # ### DIP Assignment 3
    # """)
    
    image = image_uploader()
    
    if image is not None:
        low_res_img = image.resize((128, 128))
        if st.button("Upscale Image", use_container_width=True):
            upscale_image(image=image)
            st.stop()

if __name__ == "__main__":
    main()
    
