from plate_detect import detect_plate
from char_segment import preprocess_plate, remove_plate_border, segment_characters
from char_recognize import recognize_chars


if __name__ == "__main__":
    image_path = "images/car2.jpg"

    # 第一步：车牌定位
    plate = detect_plate(image_path)

    if plate is not None:
        print("成功裁剪出车牌")

        # 第二步：车牌二值化
        binary = preprocess_plate(plate)

        # 第三步：去除边框
        no_border = remove_plate_border(binary)

        # 第四步：字符分割
        chars, regions = segment_characters(no_border)

        print("字符分割完成")

        # 第五步：字符识别
        plate_number = recognize_chars(chars)

        print("识别结果：", plate_number)

    else:
        print("车牌检测失败")