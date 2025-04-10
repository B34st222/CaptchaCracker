from captcha.image import ImageCaptcha
from random import randint, choices
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path

# Define constants
n_samples = 1000  # Number of CAPTCHA images to generate
all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # Allowed characters
train_path = Path("captchas/train")  # Directory to save generated images
train_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Initialize the CAPTCHA generator
image = ImageCaptcha(width=256, height=60)

# Generate CAPTCHA images
for _ in tqdm(range(n_samples), desc="Generating CAPTCHAs"):
    n_chars = randint(3, 10)  # Random number of characters in the CAPTCHA
    chars = "".join(choices(all_chars, k=n_chars))  # Randomly select characters
    image.write(chars, train_path / f"{chars}_{uuid4()}.png")  # Save the image