import os
import cv2
import random
import numpy as np


TRAIN_DIR = "train_chars"

# 只加入浙江车牌后 6 位字符
REAL_CHAR_CONFIGS = [
    ("output_batch/car2/chars", "A809JC"),
    ("output_batch/car3/chars", "B2FS80"),
]


def normalize_char(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img)

    if coords is None:
        return cv2.resize(img, (20, 20))

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

    return canvas


def augment_char(img):
    rows, cols = img.shape

    angle = random.uniform(-5, 5)
    scale = random.uniform(0.9, 1.1)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    aug = cv2.warpAffine(img, M, (cols, rows), borderValue=0)

    tx = random.randint(-1, 1)
    ty = random.randint(-1, 1)

    M2 = np.float32([[1, 0, tx], [0, 1, ty]])
    aug = cv2.warpAffine(aug, M2, (cols, rows), borderValue=0)

    _, aug = cv2.threshold(aug, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return aug


def add_one_plate_chars(char_dir, plate_tail):
    for i, label in enumerate(plate_tail, start=2):
        filename = f"char_{i}.jpg"
        img_path = os.path.join(char_dir, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"没有找到：{img_path}")
            continue

        normalized = normalize_char(img)

        label_dir = os.path.join(TRAIN_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        base_name = os.path.basename(os.path.dirname(char_dir))
        random_id = random.randint(10000, 99999)

        cv2.imwrite(
            os.path.join(label_dir, f"real_{base_name}_{label}_{random_id}_0.png"),
            normalized
        )

        for k in range(120):
            aug = augment_char(normalized)
            cv2.imwrite(
                os.path.join(label_dir, f"real_{base_name}_{label}_{random_id}_{k + 1}.png"),
                aug
            )

        print(f"{char_dir} 中的 {label} 已加入训练集")


def main():
    for char_dir, plate_tail in REAL_CHAR_CONFIGS:
        add_one_plate_chars(char_dir, plate_tail)

    print("浙江车牌真实字符样本添加完成！")


if __name__ == "__main__":
    main()