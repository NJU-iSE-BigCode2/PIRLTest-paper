import cv2
import numpy as np


def process(img, lower_blue, upper_blue):
    # change to hsv model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # detect white
    res = cv2.bitwise_and(img, img, mask=mask)

    img_width = res.shape[1]
    img_height = res.shape[0]
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    # gradient = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    gradient = gray
    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    # perform a series of erosion and dilation
    closed = cv2.dilate(cv2.erode(closed, None, iterations=4), None, iterations=4)

    contour, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contour, img_width, img_height

def detect_pop_up(image):
    contour, img_width, img_height = process(image, np.array([0, 0, 30]), np.array([170, 85, 45]))
    # Find black-background pop-ups.
    for c in sorted(contour, key=cv2.contourArea, reverse=True):
        # compute the rotated bounding box of the largest contour
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = max(min(Xs), 0)
        x2 = max(Xs)
        y1 = max(min(Ys), 0)
        y2 = max(Ys)
        if y1 < img_height / 30:
            continue
        height = y2 - y1
        width = x2 - x1
        if width < img_width / 4 or height < img_height / 15:
            continue
        return x1, y1, x2, y2

    # Find white-background pop-ups.
    contour, img_width, img_height = process(image, np.array([0, 0, 221]), np.array([180, 30, 255]))
    is_white = 1
    for c in sorted(contour, key=cv2.contourArea, reverse=True):
        if is_white == 0:
            break
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = max(min(Xs), 0)
        x2 = max(Xs)
        y1 = max(min(Ys), 0)
        y2 = max(Ys)
        if y1 < img_height / 30:
            continue
        height = y2 - y1
        width = x2 - x1
        if width < img_width / 4 or height < img_height / 15:
            is_white = 0
            continue
        if width > img_width / 20 * 19 and height > img_height / 3 * 2:
            is_white = 0
            continue
        return x1, y1, x2, y2
    
    return None

def main():
    import sys
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'pop-up.png'
    image = cv2.imread(image_path)
    bbox = detect_pop_up(image)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pop_up = image[y1:y2, x1:x2, :]
        cv2.imwrite(output_path, pop_up)
    else:
        print('No pop-up detected.')


if __name__ == '__main__':
    main()

