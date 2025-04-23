import os
import sys
import uuid
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, jsonify

# Add project root to path to import from the main project
webapp_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(webapp_dir)
sys.path.insert(0, project_root)

from webapp.model_manager import ModelManager

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['SAMPLE_FOLDER'] = os.path.join(app.static_folder, 'samples')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_FOLDER'], exist_ok=True)

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def index():
    """Render the main page"""
    # Get list of sample images
    sample_images = []
    for file in os.listdir(app.config['SAMPLE_FOLDER']):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_images.append(file)

    return render_template('index.html', sample_images=sample_images)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """API endpoint for image recognition"""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files and 'image_data' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        if 'image' in request.files:
            # Regular file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Generate a unique filename
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save and open the image
            file.save(filepath)
            image = Image.open(filepath).convert('L')  # Convert to grayscale

        else:
            # Base64 image data from canvas
            image_data = request.form['image_data']
            # Remove header from base64 string
            image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')

            # Save image for debugging (optional)
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

        # Get confidence method and aggregation method
        confidence_method = request.form.get('confidence_method', 'step_dependent')
        aggregation_method = request.form.get('aggregation_method', 'geometric_mean')

        # Recognize text and get confidence
        result = model_manager.recognize(image, method=confidence_method, agg_method=aggregation_method)

        # Return result
        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample/<filename>', methods=['GET'])
def get_sample(filename):
    """Return prediction for a sample image"""
    try:
        filepath = os.path.join(app.config['SAMPLE_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Sample not found'}), 404

        image = Image.open(filepath).convert('L')
        result = model_manager.recognize(image)

        # Add the sample image path
        result['image_url'] = f"/static/samples/{filename}"

        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
