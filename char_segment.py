import cv2
import os
import numpy as np


def preprocess_plate(plate_img, save_dir="output"):
    """
    对车牌图像进行灰度化、滤波、二值化处理
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 灰度化
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_dir, "05_plate_gray.jpg"), gray)

    # 2. 高斯滤波，减少噪声
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(os.path.join(save_dir, "06_plate_blur.jpg"), blur)

    # 3. Otsu 二值化
    # 因为蓝牌是深色背景、浅色字符，所以这里用 THRESH_BINARY
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(os.path.join(save_dir, "07_plate_binary_raw.jpg"), binary)

    # 4. 形态学处理，去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(os.path.join(save_dir, "08_plate_binary_clean.jpg"), binary)

    return binary


def remove_plate_border(binary_img, save_dir="output"):
    """
    去掉车牌上下左右边框，减少对字符分割的干扰
    """
    h, w = binary_img.shape

    # 根据比例裁掉边缘部分
    top = int(h * 0.08)
    bottom = int(h * 0.92)
    left = int(w * 0.03)
    right = int(w * 0.97)

    cropped = binary_img[top:bottom, left:right]

    cv2.imwrite(os.path.join(save_dir, "09_plate_no_border.jpg"), cropped)

    return cropped


def segment_characters(binary_img, save_dir="output"):
    """
    字符分割：
    1. 第一个省份汉字按固定位置裁剪
    2. 后面的字母和数字用轮廓法分割
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    h, w = binary_img.shape

    chars = []
    char_regions = []

    char_save_dir = os.path.join(save_dir, "chars")
    if not os.path.exists(char_save_dir):
        os.makedirs(char_save_dir)

    # =========================
    # 1. 先裁剪省份汉字区域
    # =========================
    province_x1 = int(w * 0.00)
    province_x2 = int(w * 0.17)
    province_y1 = int(h * 0.05)
    province_y2 = int(h * 0.95)

    province_char = binary_img[province_y1:province_y2, province_x1:province_x2]

    # 统一缩放为 20×20
    province_resized = cv2.resize(province_char, (20, 20))
    chars.append(province_resized)
    char_regions.append((province_x1, province_y1, province_x2 - province_x1, province_y2 - province_y1))

    # =========================
    # 2. 对右侧字母数字区域做轮廓分割
    # =========================
    right_start = int(w * 0.15)
    right_img = binary_img[:, right_start:]

    contours, _ = cv2.findContours(
        right_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    letter_regions = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        # 还原到原图坐标
        x = x + right_start

        area = cw * ch
        ratio = cw / ch if ch != 0 else 0

        if (
            ch > h * 0.35 and
            ch < h * 0.95 and
            cw > w * 0.015 and
            cw < w * 0.20 and
            area > 50 and
            0.15 < ratio < 1.2
        ):
            letter_regions.append((x, y, cw, ch))

    # 按 x 坐标排序
    letter_regions = sorted(letter_regions, key=lambda item: item[0])

    # 有些噪声可能会被误框，理论上蓝牌后面应该是 6 个字符
    # 如果多于 6 个，取面积较大且靠前的 6 个
    if len(letter_regions) > 6:
        letter_regions = sorted(letter_regions, key=lambda item: item[2] * item[3], reverse=True)
        letter_regions = letter_regions[:6]
        letter_regions = sorted(letter_regions, key=lambda item: item[0])

    char_regions.extend(letter_regions)

    # =========================
    # 3. 保存结果
    # =========================
    color_show = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    for idx, (x, y, cw, ch) in enumerate(char_regions):
        padding = 2
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + cw + padding, binary_img.shape[1])
        y2 = min(y + ch + padding, binary_img.shape[0])

        char_img = binary_img[y1:y2, x1:x2]
        char_resized = cv2.resize(char_img, (20, 20))

        # 重新覆盖第一个字符，保证 chars 顺序一致
        if idx == 0:
            chars[0] = char_resized
        else:
            chars.append(char_resized)

        cv2.rectangle(color_show, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(char_save_dir, f"char_{idx + 1}.jpg"), char_resized)

    cv2.imwrite(os.path.join(save_dir, "10_char_segment_result.jpg"), color_show)

    print(f"共分割出 {len(chars)} 个字符")
    return chars, char_regions