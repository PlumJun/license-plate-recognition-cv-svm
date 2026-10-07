import cv2
import numpy as np
import joblib
import os


MODEL_PATH = "models/svm_model.pkl"


def normalize_char(char_img):
    """
    将字符图像统一处理为 20×20 黑底白字图像。
    先提取白色字符外接框，再等比例缩放到 20×20 画布中，
    避免直接 resize 导致字符比例变形。
    """
    img = char_img.copy()

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(img)

    if coords is None:
        img = cv2.resize(img, (20, 20))
        return img.reshape(1, -1) / 255.0

    x, y, w, h = cv2.boundingRect(coords)
    char = img[y:y + h, x:x + w]

    target_size = 18
    ch, cw = char.shape

    if ch > cw:
        new_h = target_size
        new_w = max(1, int(cw * target_size / ch))
    else:
        new_w = target_size
        new_h = max(1, int(ch * target_size / cw))

    char = cv2.resize(char, (new_w, new_h))

    canvas = np.zeros((20, 20), dtype=np.uint8)
    x_offset = (20 - new_w) // 2
    y_offset = (20 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = char

    feature = canvas.reshape(1, -1) / 255.0
    return feature


def recognize_chars(chars):
    """
    浙江车牌识别：
    第一个字符固定为“浙”，后 6 个字符使用 SVM 识别。
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("没有找到 SVM 模型，请先运行 train_svm.py")

    model = joblib.load(MODEL_PATH)

    result = "浙"

    for idx, char_img in enumerate(chars):
        # 跳过第一个汉字区域
        if idx == 0:
            continue

        feature = normalize_char(char_img)
        pred = model.predict(feature)[0]
        result += pred

    return result