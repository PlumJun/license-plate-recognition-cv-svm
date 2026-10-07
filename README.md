# 基于传统机器视觉与 SVM 的车牌识别系统

这是一个使用 Python 编写的浙江师范大学工学院机器视觉课程项目，实现了蓝色车牌的定位、字符分割和字符识别，最后取得课程成绩95分，如果觉得对你有帮助请点个star，谢谢！车牌定位与分割主要使用 OpenCV 图像处理方法，字符分类使用 PCA 与 SVM。

## 处理流程

输入车辆图片后，程序依次执行：

1. 在 HSV 颜色空间中提取蓝色区域，并根据轮廓和宽高比筛选车牌候选区域。
2. 对车牌图像进行灰度化、高斯滤波、Otsu 二值化和形态学去噪。
3. 裁去车牌边框，并分割字符区域。
4. 使用训练好的 SVM 模型识别后 6 位字母和数字，并组合车牌号。

当前程序将第一个省份字符固定为“浙”，主要针对蓝色车牌的课程演示场景。

## 环境要求

- Python 3.9 或更高版本
- OpenCV
- NumPy
- scikit-learn
- joblib
- Pillow（用于生成训练字符图片）

## 安装与运行

在项目根目录打开终端。以下命令适用于 Windows PowerShell；直接调用虚拟环境中的 Python，可以避免 PowerShell 脚本执行策略阻止激活脚本。

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python numpy scikit-learn joblib Pillow
```

运行单张图片识别：

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` 默认读取 `images/car2.jpg`。识别过程中的中间图像和车牌裁剪结果保存在 `output/` 目录。

## 训练字符分类模型

仓库中包含 `models/svm_model.pkl`，通常可以直接运行识别。如果需要重新训练，先确认 `train_chars/` 中有按字符类别放置的训练图片，再运行：

```powershell
.\.venv\Scripts\python.exe train_svm.py
```

训练完成后，模型保存到 `models/svm_model.pkl`。如果需要生成数字和大写英文字母的合成训练图片，可以运行：

```powershell
.\.venv\Scripts\python.exe generate_train_chars.py
```

该脚本会向 `train_chars/` 写入训练样本。重新生成样本前，请留意目录中已有的数据。

## 批量识别

将待识别图片放入 `images/`，运行：

```powershell
.\.venv\Scripts\python.exe batch_test.py
```

结果保存在 `output_batch/`，其中包括每张图片的中间处理结果和 `result.csv`。脚本目前为 `car2.jpg`、`car3.jpg` 配置了示例真实车牌号用于结果对照；其他图片的真实车牌号留空。

## 项目结构

```text
.
├── images/                   # 输入示例图片
├── train_chars/              # 按类别组织的字符训练图片
├── models/
│   └── svm_model.pkl         # 训练好的字符分类模型
├── main.py                   # 单张图片识别入口
├── batch_test.py             # 批量识别入口
├── plate_detect.py           # 蓝色车牌区域定位
├── char_segment.py           # 车牌预处理与字符分割
├── char_recognize.py         # SVM 字符识别
├── train_svm.py              # PCA + SVM 模型训练
├── generate_train_chars.py   # 生成合成字符训练图片
└── add_real_chars_multi.py   # 将示例识别字符加入训练集并进行增强
```

## 当前实现的限制

- 车牌定位针对蓝色区域设计，其他颜色车牌可能无法正确定位。
- 省份字符固定输出为“浙”，没有进行汉字分类。
- 字符分割使用固定区域裁切和轮廓筛选，对倾斜、遮挡、反光或复杂背景较敏感。
- 识别效果取决于输入图像质量、训练样本和模型，不代表适用于所有车辆或场景。
