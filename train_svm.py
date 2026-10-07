import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


TRAIN_DIR = "train_chars"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")


def load_data():
    data = []
    labels = []

    for label in os.listdir(TRAIN_DIR):
        label_dir = os.path.join(TRAIN_DIR, label)

        if not os.path.isdir(label_dir):
            continue

        for filename in os.listdir(label_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (20, 20))

            # 保证黑底白字
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            feature = img.reshape(-1) / 255.0

            data.append(feature)
            labels.append(label)

    return np.array(data), np.array(labels)


def train():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    X, y = load_data()

    print("训练样本数量：", len(X))
    print("类别数量：", len(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # PCA + SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=60)),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("测试集准确率：", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)

    print("模型已保存到：", MODEL_PATH)


if __name__ == "__main__":
    train()