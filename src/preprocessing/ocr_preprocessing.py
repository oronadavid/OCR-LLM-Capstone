# preprocessing/ocr_preprocessing.py

import cv2

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def upscale_image(image, scale_x=2, scale_y=2):
    return cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

def preprocess_image(
    image_path,
    use_grayscale=True,
    use_upscale=True,
    use_blur=True,
    use_threshold=True
):
    image = cv2.imread(image_path)

    if use_grayscale:
        image = convert_to_grayscale(image)

    if use_upscale:
        image = upscale_image(image)

    if use_blur:
        image = apply_gaussian_blur(image)

    if use_threshold:
        image = apply_adaptive_threshold(image)

    return image
