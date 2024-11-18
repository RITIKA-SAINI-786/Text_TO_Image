from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Stable Diffusion model
# Set the correct data type and check for available hardware device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model with torch_dtype=torch.float32 for better precision
model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32  # Switch to float32
)
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page with the text input form

@app.route('/generate', methods=['POST'])
def generate_image():
    input_text = request.form['input_text']
    
    # Ensure that input_text is not empty
    if not input_text:
        return render_template('index.html', error="Please provide a valid input text.")
    
    try:
        # Generate an image from the input text prompt
        with torch.no_grad():
            image = model(input_text).images[0]

        # Sanitize the input text to safely use it as a filename (avoid problematic characters)
        safe_input_text = input_text.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Define the path where the generated image will be saved
        image_file_path = f'static/generated/{safe_input_text}.png'

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(image_file_path), exist_ok=True)

        # Save the generated image to the file system
        image.save(image_file_path)

        # Return the rendered HTML page with the image file path for display
        return render_template('index.html', image_file=image_file_path)

    except Exception as e:
        # Handle errors gracefully and display an error message on the webpage
        return render_template('index.html', error=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    # Ensure that the 'static/generated' directory exists for saving generated images
    os.makedirs('static/generated', exist_ok=True)

    # Run the Flask app in debug mode (ideal for development)
    app.run(debug=True)
