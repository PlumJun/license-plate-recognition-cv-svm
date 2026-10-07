import os
import csv
import cv2
from plate_detect import detect_plate
from char_segment import preprocess_plate, remove_plate_border, segment_characters
from char_recognize import recognize_chars


IMAGE_DIR = "images"
OUTPUT_DIR = "output_batch"
RESULT_CSV = os.path.join(OUTPUT_DIR, "result.csv")

GROUND_TRUTH = {
    "car2.jpg": "浙A809JC",
    "car3.jpg": "浙B2FS80",
}

def process_one_image(image_path, save_dir):
    plate = detect_plate(image_path, save_dir=save_dir)

    if plate is None:
        return "车牌定位失败"

    binary = preprocess_plate(plate, save_dir=save_dir)
    no_border = remove_plate_border(binary, save_dir=save_dir)
    chars, regions = segment_characters(no_border, save_dir=save_dir)

    if len(chars) == 0:
        return "字符分割失败"

    plate_number = recognize_chars(chars)

    return plate_number

def save_final_result_image(image_path, save_dir, plate_number):
    """
    在原图上写入最终识别结果，方便论文展示
    """
    img = cv2.imread(image_path)

    if img is None:
        return

    show_text = "Result: " + plate_number.replace("浙", "Zhe ")

    cv2.putText(
        img,
        show_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.imwrite(os.path.join(save_dir, "11_final_result.jpg"), img)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    results = []

    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGE_DIR, filename)

        save_dir = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0])
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n正在处理第 {idx} 张：{filename}")

        try:
            plate_number = process_one_image(image_path, save_dir)
        except Exception as e:
            plate_number = f"程序出错：{e}"

        print(f"识别结果：{plate_number}")
        save_final_result_image(image_path, save_dir, plate_number)

        true_number = GROUND_TRUTH.get(filename, "")
        is_correct = "是" if plate_number == true_number else "否"
        results.append([filename, true_number, plate_number, is_correct])

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图片名称", "实际车牌号", "识别车牌号", "是否正确"])
        writer.writerows(results)

    print("\n批量测试完成！")
    print(f"结果已保存到：{RESULT_CSV}")


if __name__ == "__main__":
    main()