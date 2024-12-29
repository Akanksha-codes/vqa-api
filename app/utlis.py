from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    # Resize image to match model's expected input size
    return image.resize((224, 224))
