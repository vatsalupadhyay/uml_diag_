import pytesseract
import cv2

# Custom config for Tesseract
custom_config = r'--oem 3 --psm 6'  # OEM 3 = default mode, PSM 6 = Assume a single uniform block of text

def extract_text_from_image(image_path, contours):
    img = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return []
    
    texts = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]  # Extract region of interest (shape area)
        
        # Check if the ROI is valid
        if roi.size == 0:
            print(f"Error: ROI for class {i + 1} is empty.")
            continue
        
        cv2.imwrite(f'roi_{i}.png', roi)  # Save each ROI for inspection
        text = pytesseract.image_to_string(roi, config=custom_config).strip()
        texts.append(text)
        print(f"Class {i + 1} detected text: {text}")  # Print the text for debugging
    return texts


