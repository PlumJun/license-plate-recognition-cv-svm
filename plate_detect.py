import cv2
import numpy as np
import os


def resize_image(img, max_width=1000):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_plate(image_path, save_dir="output"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("图片读取失败，请检查路径是否正确")

    img = resize_image(img)
    result_img = img.copy()

    # 1. 转 HSV，专门提取蓝色车牌区域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色车牌 HSV 范围，可根据图片微调
    lower_blue = np.array([100, 70, 70])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imwrite(os.path.join(save_dir, "01_blue_mask.jpg"), blue_mask)

    # 2. 形态学处理，让蓝色区域连成完整车牌块
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, None, iterations=2)
    morph = cv2.erode(morph, None, iterations=1)

    cv2.imwrite(os.path.join(save_dir, "02_blue_morph.jpg"), morph)

    # 3. 查找蓝色区域轮廓
    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / h

        # 蓝牌通常横向矩形，宽高比大约 2.0 ~ 5.5
        if 2.0 < ratio < 5.5 and area > 1000:
            candidates.append((x, y, w, h))

    if len(candidates) == 0:
        print("没有检测到蓝色车牌区域")
        return None

    # 4. 选择面积最大的候选区域
    candidates = sorted(candidates, key=lambda item: item[2] * item[3], reverse=True)
    x, y, w, h = candidates[0]

    # 5. 稍微扩大边界，避免裁掉车牌边缘
    padding_x = 8
    padding_y = 6

    x1 = max(x - padding_x, 0)
    y1 = max(y - padding_y, 0)
    x2 = min(x + w + padding_x, img.shape[1])
    y2 = min(y + h + padding_y, img.shape[0])

    plate = img[y1:y2, x1:x2]

    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(save_dir, "03_detect_result.jpg"), result_img)
    cv2.imwrite(os.path.join(save_dir, "04_plate.jpg"), plate)

    print("车牌定位完成，结果已保存到 output 文件夹")
    return plate