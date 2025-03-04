


from flask import Flask, request, jsonify, render_template
import os
import cv2
import requests
import zipfile
import json
from uml_to_text import extract_uml_text
from validation_rules import validate_class_diagram

app = Flask(__name__)

def clean_type(value):
    valid_types = {"int", "float", "double", "boolean", "String", "char"}
    return value if value in valid_types else "String"

def parse_extracted_text(extracted_texts):
    parsed_classes = []
    for class_text in extracted_texts:
        lines = class_text.split("\n")
        class_name = lines[0].strip() if lines else "Unknown"
        attributes = []
        methods = []
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("+"):
                if "(" in line and ")" in line:
                    methods.append(line.split("(")[0].replace("+", "").strip())
                else:
                    parts = line.replace("+", "").split(":")
                    if len(parts) == 2:
                        attr_name = parts[0].strip()
                        attr_type = clean_type(parts[1].strip())
                        attributes.append({"name": attr_name, "type": attr_type})
                    else:
                        attributes.append({"name": parts[0].strip(), "type": "String"})
        
        parsed_classes.append({"name": class_name, "attributes": attributes, "methods": methods})
    
    return {"classes": parsed_classes}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(image_path)
    print(f"✅ Image saved at: {image_path}")

    try:
        extracted_texts = extract_uml_text(image_path, debug=True)
        print("📌 Extracted Texts:", extracted_texts)
    except Exception as e:
        print(f"❌ Error during text extraction: {str(e)}")
        return jsonify({"error": "Text extraction failed."}), 500

    os.remove(image_path)
    print("🗑️ Removed uploaded image.")

    if not extracted_texts:
        return jsonify({"error": "No UML text detected."}), 400

    parsed_classes = parse_extracted_text(extracted_texts)
    print("📄 Parsed Classes:", json.dumps(parsed_classes, indent=2))

    java_api_url = "http://localhost:8080/api/generate"
    try:
        response = requests.post(java_api_url, json=parsed_classes)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Java API request failed: {str(e)}")
        return jsonify({"error": "Failed to generate Java files."}), 500

    zip_path = os.path.join("output_codes", "generated_code.zip")
    os.makedirs("output_codes", exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("output_codes")
    print("📂 Java files extracted to output_codes/")

    return jsonify({"message": "Java files generated successfully.", "output_folder": "output_codes/"})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("debug_output", exist_ok=True)
    os.makedirs("output_codes", exist_ok=True)
    print("🚀 Server started!")
    app.run(debug=True)






