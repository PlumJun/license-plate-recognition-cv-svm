import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter


SAVE_DIR = "train_chars"
IMG_SIZE = 40
FINAL_SIZE = 20
SAMPLES_PER_CLASS = 80


def get_font_path():
    """
    尝试使用 Windows 系统字体。
    如果找不到，就使用 PIL 默认字体。
    """
    font_candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ]

    for path in font_candidates:
        if os.path.exists(path):
            return path

    return None


def generate_one_char(char, font_path):
    """
    生成单个字符图片：黑底白字，接近车牌二值化后的效果
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(26, 34)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # 获取字符大小
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 随机平移
    x = (IMG_SIZE - text_w) // 2 + random.randint(-2, 2)
    y = (IMG_SIZE - text_h) // 2 + random.randint(-3, 3)

    draw.text((x, y), char, fill=255, font=font)

    # 随机旋转，模拟实际车牌倾斜
    angle = random.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    # 少量模糊
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    # 缩放为 20×20
    img = img.resize((FINAL_SIZE, FINAL_SIZE))

    # 二值化
    img = img.point(lambda p: 255 if p > 80 else 0)

    return img


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    font_path = get_font_path()
    print("使用字体：", font_path)

    # 训练数字和大写字母
    classes = list(string.digits + string.ascii_uppercase)

    for char in classes:
        char_dir = os.path.join(SAVE_DIR, char)
        os.makedirs(char_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            img = generate_one_char(char, font_path)
            img.save(os.path.join(char_dir, f"{char}_{i}.png"))

    print("训练字符生成完成！")


if __name__ == "__main__":
    main()