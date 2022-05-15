import os
import sys

import cv2
import fitz
import mimetypes
import numpy as np


def extract_page(fp: str, pn: int = None):
    if not os.path.exists(fp):
        raise "Provided file %s does not exist" % fp

    if not mimetypes.guess_type(fp)[0] == "application/pdf":
        raise "Provided file %s is not a valid PDF" % fp

    document = fitz.open(fp)
    if pn is None:
        for page in document:
            yield _pixmap_to_numpy(page.get_pixmap(matrix=fitz.Matrix(3, 3)))
    else:
        yield _pixmap_to_numpy(document.load_page(pn).get_pixmap(matrix=fitz.Matrix(3, 3)))


def extract_pdf(path: str, f: int, t: int, out: str):
    if not os.path.exists(path):
        raise ("Provided file %s does not exist" % path)

    if not mimetypes.guess_type(path)[0] == "application/pdf":
        raise ("Provided file %s is not a valid PDF" % path)

    document = fitz.open(path)

    if not os.path.exists(out):
        os.mkdir(out)

    for i in range(f, t):
        page = document.load_page(i - 1)

        table = crop_required_portions(
            _pixmap_to_numpy(page.get_pixmap(matrix=fitz.Matrix(3, 3)))
        )

        if table is not None:
            cv2.imwrite(out + "/" + str(i) + "_0.png", table[0])
            cv2.imwrite(out + "/" + str(i) + "_1.png", table[1])

        print(str(i) + "  Done")


def crop_required_portions(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(g_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    v_contours = _find_contours(thresh, (1, 10))
    h_contours = _find_contours(thresh, (10, 2))

    v_lines = sorted(
        [_max(v_contours, 3), _max(v_contours, 3), _max(v_contours, 3)],
        key=lambda x: x[0][0],
    )

    h_lines = sorted(
        [_max(h_contours, 2), _max(h_contours, 2), _max(h_contours, 2)],
        key=lambda x: x[0][1],
    )

    for lines in v_lines + h_lines:
        cv2.drawContours(img, [lines[1]], 0, (255, 255, 255), 3)

    return \
        img[h_lines[1][0][1]:h_lines[2][0][1], v_lines[0][0][0]:v_lines[2][0][0]], \
        img[h_lines[0][0][1]:h_lines[1][0][1], v_lines[1][0][0]:v_lines[2][0][0]]


def _max(_lines, _index):
    max_line = _lines[0]

    for line in _lines[1:]:
        if line[0][_index] > max_line[0][_index]:
            max_line = line

    _lines.remove(max_line)
    return max_line


def _pixmap_to_numpy(pixmap):
    im = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)
    return np.ascontiguousarray(im[..., [2, 1, 0]])


def _find_contours(_thresh, _size):
    contours = cv2.findContours(
        cv2.morphologyEx(
            _thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, _size),
            iterations=2
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = contours[0] if len(contours) == 2 else contours[1]

    lines = []
    for c in contours:
        lines.append((cv2.boundingRect(c), c))

    return lines


def main():
    if len(sys.argv) < 1:
        raise "Not enough arguments"

    if sys.argv[1] == '--version':
        print('ARKITEKT_PDF_UTIL v1.0')
        exit(0)

    if len(sys.argv) < 5:
        raise "No enough arguments, Provide filename and limit of pages"

    extract_pdf(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])


if __name__ == "__main__":
    main()

#     14 and 15
