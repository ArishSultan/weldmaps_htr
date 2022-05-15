import cv2
import pytesseract as tes
from PIL import Image, ImageFilter


def _detect_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[0], 1))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    boxes = []
    for c in cnts:
        boxes.append(cv2.boundingRect(c))

    return sorted(boxes, key=lambda x: x[1])


def _crop_line(img, contours: (int, int, int, int)):
    x, y, w, h = contours
    return img[y:y + h, x:x + w]


def _pad_line(input_image):
    size = input_image.shape
    new_size = size[1], size[0] + 20

    old_image = Image.fromarray(input_image)
    new_image = Image.new("L", new_size, 255)
    new_image.paste(old_image, (
        0,
        new_size[1] // 2 - size[0] // 2
    ))
    return new_image


def apply_ocr(img_path):
    output_boxes = []

    img = cv2.imread(img_path)

    print('DETECTING LINES ...')
    lines = _detect_lines(img)

    print('APPLYING OCR ON EACH LINE ...')
    for line in lines:
        line_img = _crop_line(img, line)

        a = _pad_line(line_img) \
            .convert('1') \
            .convert('L') \
            .filter(ImageFilter.BoxBlur(.5))

        a.resize((a.width * 2, a.height * 2))
        if line_img.shape[0] > 10:
            result = tes.image_to_data(
                a,
                output_type=tes.Output.DICT,
                config="--psm 7 -c tessedit_char_whitelist=" +
                       r"'*0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- /\*'"
            )

            if len(result) > 1:
                output_boxes.append(result)

    return output_boxes
