"""
Visit this link:
https://huggingface.co/black-forest-labs/FLUX.1-dev
Click on allow access to the model

The .env file should contain the following:
HF_TOKEN=your_hugging_face_token

Sign up for a Hugging Face account and create a new token at https://huggingface.co/settings/token
1. Click on create new token
2. Enter a name for the token
3. Click on 'Read' as the token type
3. Click on the create button


API DOC LINK: https://black-forest-labs-flux-1-schnell.hf.space/?view=api
"""

from gradio_client import Client
from PIL import Image
import dotenv

# Load the environment variables
dotenv.load_dotenv()

# Initialize the client
client = Client('black-forest-labs/FLUX.1-schnell')

# Generate the image
result = client.predict(
    prompt="Cats on dogs",  # The prompt text should go over here.
    seed=0,
    randomize_seed=True,
    width=512,
    height=512,
    num_inference_steps=4,
    api_name='/infer'
)

# The result is a tuple with the image path and a number
# NOTE: your generated image is stored locally in the path provided in the image_path
image_path, _ = result

# Open and display the image
image = Image.open(image_path)

# The image is of the type PIL
image = image.convert('RGB')
