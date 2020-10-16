import argparse
import cv2
import glob
import imutils
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
import tqdm
from core.callbacks import TrainingMonitor
from core.nn import LeNet
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    args = parser.parse_args()

    data = []
    labels = []

    print("[INFO] Loading dataset ...")
    img_paths = glob.glob(f"{args.dataset}/*/*/*.jpg")
    for img_path in tqdm.tqdm(img_paths):
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28)
        image = img_to_array(image)
        data.append(image)
        # Read label
        label = img_path.split(os.path.sep)[-3]
        labels.append(label)

    # Standardize data
    data = np.array(data, dtype="float") / 255.

    # One hot encoding
    labels = np.array(labels)
    label_encoder = LabelEncoder().fit(labels)
    labels = to_categorical(label_encoder.transform(labels), 2)
    print(f"[INFO] Data: {data.shape}. Labels: {labels.shape}")

    # Handle class imbalance
    class_total = labels.sum(axis=0)
    class_weight = class_total.max() / class_total
    class_weight = { i: class_weight[i] for i in range(len(class_weight)) }
    print(f"[INFO] Class weights: {class_weight}")

    print("[INFO] Train/Test split ...")
    # Split train test set based on y
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"Training: {X_train.shape}. Testing: {X_test.shape}")

    print("[INFO] Compiling model ...")
    model = LeNet.build(height=28, width=28, channels=1, classes=2)
    optimizer = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Create callbacks
    os.makedirs("outputs", exist_ok=True)
    fig_path = f"outputs/{os.getpid()}.png"
    json_path = f"outputs/{os.getpid()}.json"
    training_monitor = TrainingMonitor(fig_path, json_path=json_path)

    checkpoint = ModelCheckpoint("weights.hdf5", monitor="val_loss", mode="min", save_best_only=True, verbose=1)

    callbacks = [training_monitor, checkpoint]

    print("[INFO] Training model ...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=class_weight, batch_size=64, epochs=25, callbacks=callbacks, verbose=1)

    print("[INFO] Evaluate model ...")
    model = load_model("weights.hdf5")
    preds = model.predict(X_test, batch_size=64)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=label_encoder.classes_)
    print(report)


if __name__ == '__main__':
    main()
