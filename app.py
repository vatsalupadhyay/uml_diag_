from flask import Flask, request, jsonify, render_template
import os
import cv2
from parser import preprocess_image, detect_shapes
from ocr import extract_text_from_image
from validation_rules import validate_class_diagram

app = Flask(__name__)

@app.route('/')
def home():
    # Render the main HTML page with an upload form
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']

    # Check if an image was uploaded
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # Preprocess the image and detect shapes
    preprocessed_img = preprocess_image(image_path)
    shapes = detect_shapes(preprocessed_img)

    # Extract text from the detected shapes
    texts = extract_text_from_image(image_path, shapes)

    # Validate the UML class diagram
    validation_result = validate_class_diagram(shapes, texts)

    # Delete the temporary image file after processing
    os.remove(image_path)

    # Send the result back as JSON
    return jsonify({"results": validation_result})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Create uploads folder if it doesn't exist
    app.run(debug=True)
