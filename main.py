from parser import preprocess_image, detect_shapes
from ocr import extract_text_from_image
from validation_rules import validate_class_diagram
import sys
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def main():
    # Check if the user passed an image file as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        return
    
    image_path = sys.argv[1]

    # Preprocess the image (convert to grayscale and apply edge detection)
    preprocessed_img = preprocess_image(image_path)
    
    # Detect shapes (rectangles representing UML classes)
    shapes = detect_shapes(preprocessed_img)
    
    # Extract text from the detected shapes
    texts = extract_text_from_image(image_path, shapes)
    
    # Validate the UML class diagram
    validation_result = validate_class_diagram(shapes, texts)
    
    # Display the result
    for result in validation_result:
        print(result)

if __name__ == "__main__":
    main()
