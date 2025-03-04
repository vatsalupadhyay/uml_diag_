
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def preprocess_image(image_path, debug=True):
    """
    Advanced preprocessing with multiple techniques to improve text clarity
    """
    # Read image
    img = cv2.imread(image_path)
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove shadows and normalize lighting
    dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        norm_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Remove noise while preserving text edges
    denoise = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    if debug:
        debug_save_image(original, "1_original")
        debug_save_image(norm_img, "2_normalized")
        debug_save_image(thresh, "3_threshold")
        debug_save_image(denoise, "4_denoised")
    
    return denoise

def enhance_text_region(roi, debug=True):
    """
    Advanced text region enhancement with multiple scaling attempts
    """
    # Try multiple scale factors
    scale_factors = [2.0, 3.0, 4.0]
    best_text = ""
    best_confidence = 0
    
    for scale in scale_factors:
        # Scale the image
        scaled = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale if needed
        if len(scaled.shape) == 3:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        preprocessed_versions = []
        
        # Version 1: Basic thresholding
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_versions.append(binary)
        
        # Version 2: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_versions.append(adaptive)
        
        # Version 3: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(scaled)
        preprocessed_versions.append(contrast_enhanced)
        
        # Try OCR on each version
        for idx, processed in enumerate(preprocessed_versions):
            # Configure Tesseract for better accuracy
            custom_config = f'''--oem 1 --psm 6 
                -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789():,.-_+
                -c tessedit_do_invert=0
                -c textord_heavy_nr=1
                -c textord_min_linesize=1'''
            
            # Get text and confidence scores
            text = pytesseract.image_to_string(processed, config=custom_config)
            confidence_data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence for this version
            confidences = [float(x) for x in confidence_data['conf'] if x != '-1']
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = text
                    
                    if debug:
                        debug_save_image(processed, f"5_best_text_scale_{scale}_version_{idx}")
    
    return best_text

def detect_text_regions(preprocessed_img, original_img, debug=True):
    """
    Improved text region detection using connected components and contours
    """
    # Find connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(preprocessed_img), connectivity=8)
    
    # Find contours
    contours, _ = cv2.findContours(cv2.bitwise_not(preprocessed_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine both approaches
    text_regions = []
    debug_img = original_img.copy()
    
    # Process connected components
    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area > 1000 and w/h < 3 and h/w < 3:  # Adjusted ratios
            text_regions.append((x, y, w, h))
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if debug:
        debug_save_image(debug_img, "6_text_regions")
    
    return text_regions

def extract_text_from_region(img, region, debug=True):
    """
    Extract text from a region using multiple OCR attempts
    """
    x, y, w, h = region
    roi = img[y:y+h, x:x+w]
    
    # Get enhanced text
    text = enhance_text_region(roi, debug)
    
    # Basic cleaning
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    return text

def extract_uml_text(image_path, debug=True):
    """
    Main function to extract text from UML diagram
    """
    # Preprocess image
    preprocessed = preprocess_image(image_path, debug)
    original = cv2.imread(image_path)
    
    # Detect text regions
    regions = detect_text_regions(preprocessed, original, debug)
    
    # Extract text from each region
    extracted_texts = []
    for i, region in enumerate(regions, 1):
        text = extract_text_from_region(preprocessed, region, debug)
        if text:
            extracted_texts.append(f"Block {i}:\n{text}")
    
    return extracted_texts

def debug_save_image(img, name, debug_dir="debug_output"):
    """
    Save debug images
    """
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)

# def main():
#     image_path = "D:/uml_py/uml_diag_/images/uml4.png"
#     debug = True
    
#     try:
#         texts = extract_uml_text(image_path, debug)
        
#         print("Extracted Text Blocks:")
#         print("----------------------")
#         for text in texts:
#             print(f"\n{text}")
#             print("----------------------")
            
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()



# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import os
# import re

# def preprocess_image(image_path, debug=True):
#     """
#     Enhanced preprocessing with better text preservation
#     """
#     img = cv2.imread(image_path)
#     original = img.copy()
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Enhance contrast
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     contrast_enhanced = clahe.apply(gray)
    
#     # Remove noise while preserving text
#     denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
    
#     # Adaptive thresholding
#     thresh = cv2.adaptiveThreshold(
#         denoised,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11,
#         2
#     )
    
#     if debug:
#         debug_save_image(original, "1_original")
#         debug_save_image(contrast_enhanced, "2_contrast")
#         debug_save_image(thresh, "3_threshold")
    
#     return thresh, original

# def detect_class_boxes(img, debug=True):
#     """
#     Detect class boxes and their hierarchical structure
#     """
#     # Find all contours
#     contours, hierarchy = cv2.findContours(
#         cv2.bitwise_not(img),
#         cv2.RETR_TREE,
#         cv2.CHAIN_APPROX_SIMPLE
#     )
    
#     class_boxes = []
#     debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)
#         aspect_ratio = w/h
        
#         # Filter valid class boxes
#         if area > 1000 and 0.5 < aspect_ratio < 2.0:
#             # Get the top portion (potential class name)
#             top_region = (x, y, w, min(30, h))
#             # Get the main content region
#             main_region = (x, y + min(30, h), w, h - min(30, h))
            
#             class_boxes.append({
#                 'top': top_region,
#                 'main': main_region,
#                 'full': (x, y, w, h)
#             })
            
#             if debug:
#                 # Draw rectangles for visualization
#                 cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.rectangle(debug_img, (x, y), (x+w, y+min(30, h)), (255, 0, 0), 2)
    
#     if debug:
#         debug_save_image(debug_img, "4_detected_boxes")
    
#     return class_boxes

# def enhance_text_region(roi, debug=True):
#     """
#     Enhanced text region processing with multiple attempts
#     """
#     # Scale up the image
#     scale_factor = 3
#     scaled = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
#     if len(scaled.shape) == 3:
#         scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    
#     # Try multiple preprocessing techniques
#     versions = []
    
#     # Version 1: Basic threshold
#     _, v1 = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     versions.append(v1)
    
#     # Version 2: Adaptive threshold
#     v2 = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     versions.append(v2)
    
#     # Version 3: Enhanced contrast
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     v3 = clahe.apply(scaled)
#     versions.append(v3)
    
#     best_text = ""
#     best_confidence = 0
    
#     for idx, version in enumerate(versions):
#         custom_config = r'''--oem 1 --psm 6 
#             -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789():,.-_+
#             -c tessedit_do_invert=0'''
        
#         text = pytesseract.image_to_string(version, config=custom_config)
#         conf_data = pytesseract.image_to_data(version, config=custom_config, output_type=pytesseract.Output.DICT)
        
#         # Calculate confidence
#         confidences = [float(x) for x in conf_data['conf'] if x != '-1']
#         if confidences:
#             avg_confidence = sum(confidences) / len(confidences)
#             if avg_confidence > best_confidence:
#                 best_confidence = avg_confidence
#                 best_text = text
                
#                 if debug:
#                     debug_save_image(version, f"5_text_version_{idx}")
    
#     return best_text.strip()

# def clean_text(text):
#     """
#     Clean and format extracted text
#     """
#     # Basic cleaning
#     text = text.replace('\n\n', '\n').strip()
#     text = re.sub(r'\s+', ' ', text)
    
#     # Fix common OCR errors
#     text = text.replace('Iur', 'Int')
#     text = text.replace('Tat', 'Int')
#     text = text.replace('productld', 'productId')
#     text = text.replace('customerld', 'customerId')
#     text = text.replace('orderld', 'orderId')
    
#     return text

# def extract_uml_class(img, class_box, debug=True):
#     """
#     Extract class name and content separately
#     """
#     x, y, w, h = class_box['full']
#     tx, ty, tw, th = class_box['top']
#     mx, my, mw, mh = class_box['main']
    
#     # Extract class name from top region
#     top_roi = img[ty:ty+th, tx:tx+tw]
#     class_name = enhance_text_region(top_roi, debug)
    
#     # Extract main content
#     main_roi = img[my:my+mh, mx:mx+mw]
#     content = enhance_text_region(main_roi, debug)
    
#     return clean_text(class_name), clean_text(content)

# def extract_uml_text(image_path, debug=True):
#     """
#     Main function to extract UML class diagram text
#     """
#     preprocessed, original = preprocess_image(image_path, debug)
#     class_boxes = detect_class_boxes(preprocessed, debug)
    
#     extracted_classes = []
#     for i, box in enumerate(class_boxes, 1):
#         class_name, content = extract_uml_class(preprocessed, box, debug)
        
#         # Format the output
#         class_text = f"Class: {class_name if class_name else f'Class {i}'}\n"
#         class_text += "------------------------\n"
        
#         # Split content into attributes and methods
#         lines = content.split('\n')
#         attributes = []
#         methods = []
        
#         for line in lines:
#             line = line.strip()
#             if line.startswith('+'):
#                 methods.append(line)
#             elif line.startswith('-'):
#                 attributes.append(line)
#             elif line:  # Handle lines without +/- prefix
#                 if '(' in line:
#                     methods.append(f"+ {line}")
#                 else:
#                     attributes.append(f"- {line}")
        
#         if attributes:
#             class_text += "Attributes:\n"
#             class_text += '\n'.join(attributes) + '\n'
        
#         if methods:
#             class_text += "\nMethods:\n"
#             class_text += '\n'.join(methods)
        
#         extracted_classes.append(class_text)
    
#     return extracted_classes

# def debug_save_image(img, name, debug_dir="debug_output"):
#     if not os.path.exists(debug_dir):
#         os.makedirs(debug_dir)
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img)

# def main():
#     image_path = "D:/uml_py/uml_diag_/images/uml3.png"
#     debug = True
    
#     try:
#         classes = extract_uml_text(image_path, debug)
        
#         print("Extracted UML Class Diagram:")
#         print("============================")
#         for class_text in classes:
#             print(f"\n{class_text}")
#             print("----------------------------")
            
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()