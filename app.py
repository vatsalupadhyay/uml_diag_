import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import pytesseract
from uml_processor import UMLProcessor
from java_generator import JavaCodeGenerator
import re
import requests
import zipfile
import json
from uml_to_text import extract_uml_text
from validation_rules import validate_class_diagram
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_class_box(contour, img_shape):
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    img_area = img_shape[0] * img_shape[1]

    if area < 20000 or area > img_area * 0.5:
        return False

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) < 4 or len(approx) > 8:
        return False

    aspect_ratio = float(w) / h
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False

    return True

def remove_nested_boxes(boxes, overlap_threshold=0.7):
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    result = []

    for i, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        is_nested = False

        for j, box2 in enumerate(result):
            x2, y2, w2, h2 = box2

            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap

            area1 = w1 * h1
            iou = intersection / area1

            if iou > overlap_threshold:
                is_nested = True
                break

        if not is_nested:
            result.append(box1)

    return result

def detect_yellow_headers(img, boxes):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = []
    for box in boxes:
        x, y, w, h = box
        top_area = yellow_mask[y:y+int(h*0.3), x:x+w]
        yellow_pixels = cv2.countNonZero(top_area)

        if yellow_pixels > (w * int(h*0.3) * 0.1):
            result.append(box)
        elif 0.5 < float(w)/h < 2.0:
            result.append(box)

    return result

def clean_class_name(name):
    name = re.sub(r'^[+\-\s]+', '', name)
    name = re.sub(r'^[0-9\W]+', '', name)
    name = re.sub(r'[\W]+$', '', name)
    return name.strip()

def extract_class_info(img, box):
    x, y, w, h = box

    padding = 5
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
    roi = img[y1:y2, x1:x2]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(roi_binary) < 127:
        roi_binary = cv2.bitwise_not(roi_binary)

    configs = [
        '--oem 3 --psm 6',  
        '--oem 3 --psm 3',  
        '--oem 3 --psm 4'   
    ]
    
    best_text = ""
    for config in configs:
        text = pytesseract.image_to_string(roi_binary, config=config)
        if len(text.strip()) > len(best_text.strip()):
            best_text = text

    lines = [line.strip() for line in best_text.strip().split('\n') if line.strip()]
    
    print(f"\nProcessing box {x},{y},{w},{h}:")
    print("Raw OCR text:")
    print(best_text)
    print("Processed lines:")
    print(lines)

    common_class_names = ['Customer', 'Order', 'Stock', 'Product', 'User', 'Account', 'Transaction']
    
    class_name = None
    for line in lines:
        # Clean the line
        clean_line = re.sub(r'^[+\-~]', '', line)  
        clean_line = re.sub(r'\s*:\s*\w+$', '', clean_line)  
        clean_line = re.sub(r'[^\w\s]', '', clean_line)  
        clean_line = clean_line.strip()
        
        for common_name in common_class_names:
            if common_name.lower() in clean_line.lower():
                class_name = common_name
                break
        
        if not class_name and clean_line:
            if (clean_line[0].isupper() and 
                len(clean_line) > 2 and 
                not any(c.isdigit() for c in clean_line)):
                class_name = clean_line
                break

    if not class_name:
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    attr_name = parts[0].strip()
                    attr_name = re.sub(r'^[+\-~]', '', attr_name)
                    if (attr_name and 
                        attr_name[0].isupper() and 
                        len(attr_name) > 2 and 
                        not any(c.isdigit() for c in attr_name)):
                        class_name = attr_name
                        break

    if not class_name and lines:
        for line in lines:
            clean_line = re.sub(r'^[+\-~]', '', line)
            clean_line = re.sub(r'[^\w\s]', '', clean_line)
            if clean_line and clean_line[0].isupper():
                class_name = clean_line
                break

    if class_name:
        class_name = clean_class_name(class_name)
        print(f"Detected class name: {class_name}")
    else:
        class_name = "Unknown"
        print("Warning: Could not detect class name")

    attributes = []
    methods = []

    for line in lines:
        line = line.strip()
        if not line or line == class_name:
            continue

        if any(marker in line for marker in ['0..%', '1..%', '*']):
            continue

        if '(' in line and ')' in line:
            method = line.split('(')[0].strip()
            method = re.sub(r'^[+\-~]', '', method)
            method = re.sub(r'[^\w\s]', '', method)
            if method:
                methods.append(method)
                print(f"Detected method: {method}")
        
        elif ':' in line:
            attr = line.strip()
            attr = re.sub(r'^[+\-~]', '', attr)
            if ':' in attr:
                attr_name, attr_type = attr.split(':', 1)
                attr_name = attr_name.strip()
                attr_type = attr_type.strip()
                
                attr_type = re.sub(r'[^\w]', '', attr_type)
                if not attr_type:
                    attr_type = 'String'
                
                attr_name = re.sub(r'[^\w]', '', attr_name)
                
                if attr_name and attr_type:
                    attributes.append(f"{attr_name}:{attr_type}")
                    print(f"Detected attribute: {attr_name}:{attr_type}")

    attributes = list(dict.fromkeys(attributes))
    methods = list(dict.fromkeys(methods))

    return {
        'class_name': class_name,
        'attributes': attributes,
        'methods': methods,
        'full_text': best_text
    }

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    scale_factor = 2
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    edges = cv2.Canny(thresh, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    possible_class_boxes = []
    for contour in contours:
        if is_class_box(contour, img.shape):
            x, y, w, h = cv2.boundingRect(contour)
            possible_class_boxes.append((x, y, w, h))
    
    filtered_boxes = remove_nested_boxes(possible_class_boxes)
    
    if len(filtered_boxes) < 3:
        final_boxes = filtered_boxes
    else:
        final_boxes = detect_yellow_headers(img, filtered_boxes)
    
    return img, final_boxes, scale_factor

def extract_text_from_image(image_path):
    try:
        processed_image, boxes, scale_factor = preprocess_image(image_path)
        
        class_info = []
        for i, box in enumerate(boxes):
            info = extract_class_info(processed_image, box)
            info['id'] = i + 1
            info['position'] = (box[0]//scale_factor, box[1]//scale_factor, 
                              box[2]//scale_factor, box[3]//scale_factor)
            
            class_info.append(info)
        
        return class_info
        
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        raise

def generate_java_code(scxml_content):
    generator = JavaCodeGenerator(scxml_content)
    return generator.generate()

def clean_type(type_str):
    type_mapping = {
        'Int': 'Integer',
        'INTEGER': 'Integer',
        'int': 'Integer',
        '_DateTime': 'LocalDateTime',
        'DateTime': 'LocalDateTime',
        'Float': 'Float',
        'float': 'Float',
        'String': 'String',
        'Bool': 'Boolean',
        'boolean': 'Boolean',
        'Date': 'LocalDate'
    }
    
    cleaned_type = re.sub(r'[^\w]', '', type_str).title()
    return type_mapping.get(cleaned_type, type_mapping.get(cleaned_type.title(), 'String'))

def clean_variable_name(name):
    name_fixes = {
        'productld': 'productId',
        'orderld': 'orderId',
        'customerld': 'customerId',
        'shopld': 'shopId',
        'arderid': 'orderId',
        'productid': 'productId',
        'AddStock': 'addStock',
        'SelectStockltem': 'selectStockItem',
        'Stockltem': 'stockItem'
    }
    
    cleaned_name = re.sub(r'[^\w]', '', name)
    
    if cleaned_name in name_fixes:
        return name_fixes[cleaned_name]
    
    if cleaned_name and cleaned_name[0].isupper():
        cleaned_name = cleaned_name[0].lower() + cleaned_name[1:]
    
    return cleaned_name

def generate_java_imports(attributes, methods):
    imports = set()
    
    for attr in attributes:
        if ':' in attr:
            attr_type = attr.split(':')[1].strip()
            if 'DateTime' in attr_type or '_DateTime' in attr_type:
                imports.add('java.time.LocalDateTime')
            elif 'Date' in attr_type:
                imports.add('java.time.LocalDate')
            elif 'Time' in attr_type:
                imports.add('java.time.LocalTime')
            elif 'List' in attr_type:
                imports.add('java.util.List')
            elif 'Set' in attr_type:
                imports.add('java.util.Set')
            elif 'Map' in attr_type:
                imports.add('java.util.Map')
    
    imports.add('java.util.Scanner')
    
    return sorted(list(imports))

def generate_main_method(class_name, attributes, methods):
    main_method = f"""
    public static void main(String[] args) {{
        {class_name} {class_name.lower()} = new {class_name}();
        System.out.println("Created new {class_name} instance.");
    }}
"""
    return main_method

def parse_extracted_text(extracted_texts):
    parsed_classes = []
    for text_block in extracted_texts:
        lines = text_block.split("\n")[1:] if "Block" in text_block else text_block.split("\n")
        class_name = None
        attributes = []
        methods = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if not line.startswith('+') and not line.startswith('-'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        attr_name = parts[0].strip()
                        attr_type = clean_type(parts[1].strip())
                        attributes.append({"name": attr_name, "type": attr_type})
            
            elif line.startswith('+'):
                method_text = line.replace('+', '').strip()
                method_text = method_text.replace('Q)', '()')  
                method_text = method_text.replace(')', '()')   
                methods.append(method_text)
        
        if not class_name and attributes:
            class_name = attributes[0]["name"].replace("id", "").capitalize()
        
        if class_name:
            parsed_classes.append({
                "name": class_name,
                "attributes": attributes,
                "methods": methods
            })
    
    return {"classes": parsed_classes}

@app.route('/')
def home():
    return render_template('index.html', 
                         title='UML to Java Code Generator',
                         description='Upload a UML class diagram to generate Java code.')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    image_path = os.path.join("uploads", secure_filename(file.filename))
    os.makedirs("uploads", exist_ok=True)
    file.save(image_path)
    print(f"✅ Image saved at: {image_path}")

    try:
        class_info = extract_text_from_image(image_path)
        class_info = [info for info in class_info if info['class_name'] != 'Unknown']
        
        if not class_info:
            return jsonify({"error": "No valid classes detected in the image."}), 400

        print("\n📌 Extracted Class Information:")
        for info in class_info:
            print(f"\nClass: {info['class_name']}")
            print("Attributes:")
            for attr in info['attributes']:
                print(f"  {attr}")
            print("Methods:")
            for method in info['methods']:
                print(f"  {method}")
    except Exception as e:
        print(f"❌ Error during text extraction: {str(e)}")
        return jsonify({"error": "Text extraction failed."}), 500

    os.remove(image_path)
    print("🗑️ Removed uploaded image.")

    try:
        output_dir = os.path.join("output_codes", "generated")
        os.makedirs(output_dir, exist_ok=True)

        generated_files = []
        for class_data in class_info:
            class_name = class_data['class_name']
            file_path = os.path.join(output_dir, f"{class_name}.java")
            
            with open(file_path, "w") as f:
                imports = generate_java_imports(class_data['attributes'], class_data['methods'])
                for import_stmt in imports:
                    f.write(f"import {import_stmt};\n")
                if imports:
                    f.write("\n")

                f.write(f"public class {class_name} {{\n\n")
                
                for attr in class_data['attributes']:
                    if ':' in attr:
                        attr_name, attr_type = attr.split(':', 1)
                        attr_name = clean_variable_name(attr_name.strip())
                        attr_type = clean_type(attr_type.strip())
                        f.write(f"    private {attr_type} {attr_name};\n")
                    else:
                        attr_name = clean_variable_name(attr)
                        if 'id' in attr_name.lower():
                            f.write(f"    private Integer {attr_name};\n")
                        else:
                            f.write(f"    private String {attr_name};\n")

                f.write(f"\n    public {class_name}() {{\n    }}\n\n")
                
                for attr in class_data['attributes']:
                    if ':' in attr:
                        attr_name, attr_type = attr.split(':', 1)
                        attr_name = clean_variable_name(attr_name.strip())
                        attr_type = clean_type(attr_type.strip())
                    else:
                        attr_name = clean_variable_name(attr)
                        attr_type = 'Integer' if 'id' in attr_name.lower() else 'String'
                    
                    method_name = attr_name[0].upper() + attr_name[1:]
                    
                    f.write(f"    public {attr_type} get{method_name}() {{\n")
                    f.write(f"        return {attr_name};\n")
                    f.write("    }\n\n")
                    
                    f.write(f"    public void set{method_name}({attr_type} {attr_name}) {{\n")
                    f.write(f"        this.{attr_name} = {attr_name};\n")
                    f.write("    }\n\n")
                
                for method in class_data['methods']:
                    clean_method = clean_variable_name(method)
                    f.write(f"    public void {clean_method}() {{\n")
                    f.write("        // TODO: Implement method\n")
                    f.write(f"        System.out.println(\"Executing {clean_method}...\");\n")
                    f.write("    }\n\n")
                
                main_method = generate_main_method(class_name, class_data['attributes'], class_data['methods'])
                f.write(main_method)
                
                f.write("}\n")
            
            generated_files.append(class_name)
            print(f"✅ Generated {class_name}.java")
        
        return jsonify({
            "message": "Java files generated successfully.",
            "output_folder": output_dir,
            "total_classes": len(class_info),
            "classes": [{
                "name": info['class_name'],
                "attributes": [{"name": attr.split(':')[0].strip(), "type": attr.split(':')[1].strip()} 
                             for attr in info['attributes'] if ':' in attr],
                "methods": info['methods']
            } for info in class_info],
            "generated_files": generated_files
        })

    except Exception as e:
        print(f"❌ Error generating Java files: {str(e)}")
        return jsonify({"error": "Failed to generate Java files."}), 500

@app.route('/output_codes/generated/<filename>')
def serve_file(filename):
    return send_from_directory('output_codes/generated', filename)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("debug_output", exist_ok=True)
    os.makedirs("output_codes", exist_ok=True)
    print("🚀 Server started!")
    app.run(debug=True)
 