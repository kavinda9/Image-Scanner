import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from flask import Flask, render_template, request, jsonify
import cv2
from werkzeug.utils import secure_filename

from utils.preprocessing import preprocess_image
from utils.ocr import text_from_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Read image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'success': False, 'error': 'Could not read image'}), 400

        # Preprocess
        preprocessed = preprocess_image(img)
        
        # Extract text
        extracted_text = text_from_image(preprocessed)
        
        # Encode preprocessed image to base64 for real-time frontend debugging
        import base64
        _, buffer = cv2.imencode('.png', preprocessed)
        preprocessed_base64 = base64.b64encode(buffer).decode('utf-8')
        preprocessed_src = f"data:image/png;base64,{preprocessed_base64}"
        
        return jsonify({
            'success': True,
            'text': extracted_text,
            'preprocessed_img': preprocessed_src,
            'message': 'Text extracted successfully'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    print("🚀 Prescription OCR Web App Running...")
    print("Open browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)