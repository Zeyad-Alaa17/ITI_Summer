import cv2
import numpy as np


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters[0]
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def display_lines(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = [np.average(left_fit, axis=0)]
    right_fit_average = [np.average(right_fit, axis=0)]
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    averaged_lines = np.array([left_line, right_line])
    line_image = np.zeros_like(img)
    if averaged_lines is not None:
        for x1, y1, x2, y2 in averaged_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def enhance(img):

    img_blur = cv2.bilateralFilter(img, 5, 75, 75)
    ret, thresh = cv2.threshold(img_blur, 153, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, (3, 3), iterations=3)
    return dilated


def get_lanes(image, points=None, rho=1, theta=np.pi / 180, threshold=15,
              minLineLength=40, maxLineGap=100):
    if points is None:
        y = image.shape[0]
        x = image.shape[1]
        points = [(.1 * x, y), (.42 * x, .65 * y), (.6 * x, .65 * y), (image.shape[1], image.shape[0])]

    vertices = np.array([points], dtype=np.int32)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img_gray, mask)
    thresh = enhance(masked_img)
    lines = cv2.HoughLinesP(thresh, rho, theta, threshold, np.array([]), minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    print(lines)
    if lines is not None:
        line_image = display_lines(image, lines)
        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combo_image
    else:
        st.warning("No lanes detected!")


from PIL import Image
import streamlit as st

uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    i = cv2.imread('image-2.jpeg')
    print(image == i)
    dim = (853, 480)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    lanes = get_lanes(resized)
    if lanes is not None:
        st.image(cv2.cvtColor(lanes, cv2.COLOR_RGB2BGR))
