import re

import Image
import cv2
import numpy as np

import pytesseract as pytesseract
import os

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
origin_path = path + '/origin_path/'

i = 0
for image in os.listdir(origin_path)[:239]:
    print(image.split(".jpg")[0])
    img = cv2.imread(origin_path + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    # 固定阈值二值化
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # 自适应二值化
    # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    cv2.imshow("binary", binary)

    # 形态学的处理，滤除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilate_image = cv2.dilate(binary, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_image = cv2.erode(dilate_image, kernel)
    cv2.imshow("erode_image", erode_image)

    # 将dilate_image转为Image
    textImage = Image.fromarray(erode_image)
    # 识别
    char = pytesseract.image_to_string(textImage, lang='engnum', config='--psm 10 --tessdata-dir tessdata')
    char = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", char)  # 去除识别出来的特殊字符
    char = char[0:4]  # 只获取前4个字符
    print(char)
    if image.split(".jpg")[0] == char:
        i += 1
print("成功识别的图片总数: %s" % i)
