import cv2

def preprocess_image(image_path):
    # Load the image and convert it to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to make text clearer for OCR
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply adaptive thresholding for better contrast
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Use edge detection (optional)
    edges = cv2.Canny(binary_img, 50, 150)
    return binary_img


def detect_shapes(preprocessed_image):
    # Detect contours which correspond to shapes
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # If the contour is a rectangle
            shapes.append(contour)
    return shapes
