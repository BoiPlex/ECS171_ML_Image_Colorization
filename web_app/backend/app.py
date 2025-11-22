"""
Flask Backend for Image Colorization
"""

from flask import Flask, send_from_directory, request, jsonify
from PIL import Image
import torch
import numpy as np
import io
import os

# Model constants
MODEL_WIDTH = 28
MODEL_HEIGHT = 28

# Import model from backend package
from model import ColorizationModel

# Get the path to the frontend folder
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__, static_folder=frontend_path, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load model on startup (placeholder)
model = ColorizationModel()
model.eval()
print("ColorizationModel instantiated from backend/model.py")


def preprocess_image(image_pil):
    """
    Preprocess image for colorization model.
    
    1. Resize to model input size
    2. Convert to grayscale
    3. Convert to tensor
    4. Normalize
    
    Args:
        image_pil: PIL Image
        
    Returns:
        torch.Tensor: Preprocessed image
    """
    # Resize to model input size
    image_resized = image_pil.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to grayscale
    image_gray = image_resized.convert('L')
    
    # Convert to numpy array
    image_array = np.array(image_gray, dtype=np.float32) / 255.0
    
    # Convert to tensor (add channel and batch dimensions)
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return tensor


def postprocess_output(tensor_output):
    """
    Convert model output tensor to PIL Image.
    
    Args:
        tensor_output: torch.Tensor of shape (1, 3, H, W)
        
    Returns:
        PIL.Image: RGB image
    """
    # Detach and move to CPU
    output = tensor_output.detach().cpu()
    
    # Remove batch dimension
    output = output.squeeze(0)
    
    # Transpose to (H, W, C)
    output = output.permute(1, 2, 0)
    
    # Convert to numpy and clip to [0, 1]
    output_array = output.numpy()
    output_array = np.clip(output_array, 0, 1)
    
    # Convert to [0, 255] and uint8
    output_array = (output_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(output_array, mode='RGB')
    
    return image_pil


@app.route('/')
def index():
    """Serve home page"""
    return send_from_directory(frontend_path, 'project.html')


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files from frontend folder"""
    return send_from_directory(frontend_path, filename)


@app.route('/colorize', methods=['POST'])
def colorize():
    """
    Colorize an image.
    
    Accepts: image file (any size, any color format)
    Returns: colorized RGB image
    """
    # Check if image file is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Open image
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess image
    tensor_input = preprocess_image(image)
    
    # Run through model
    with torch.no_grad():
        tensor_output = model(tensor_input)
    
    # Postprocess output
    colorized_image = postprocess_output(tensor_output)
    
    # Save to bytes and return
    img_bytes = io.BytesIO()
    colorized_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'model_size': (MODEL_HEIGHT, MODEL_WIDTH)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
