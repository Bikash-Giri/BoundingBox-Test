import os
import random
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint


IMAGE_SIZE = (64, 64)
DATA_DIR = 'outputs'
CSV_NAME = 'train.csv'
MODEL_PATH = 'my_model.h5'
CLASSES_PATH = 'label_classes.npy'


def load_dataframe(data_dir: str, csv_name: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'Patch' not in df.columns or 'PatchLabel' not in df.columns:
        raise KeyError("CSV must contain 'Patch' and 'PatchLabel' columns")
    return df


def read_image_safe(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0
    return img


def load_images(paths: list[str]) -> np.ndarray:
    images: list[np.ndarray] = []
    missing: list[str] = []
    for p in paths:
        img = read_image_safe(p)
        if img is None:
            missing.append(p)
            continue
        images.append(img)
    if missing:
        print(f"Warning: {len(missing)} images missing or unreadable. Example: {missing[:3]}")
    if not images:
        raise RuntimeError("No images were loaded. Check file paths in CSV.")
    return np.stack(images)


def predict_image(model: Sequential, image: np.ndarray, label_encoder: LabelEncoder) -> tuple[str, float]:
    """
    Predict the class label for a single preprocessed image.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array (64, 64, 3) normalized to [0, 1]
        label_encoder: LabelEncoder used during training
        
    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
    pred_class = int(np.argmax(pred, axis=1)[0])
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    confidence = float(pred[0][pred_class])
    return pred_label, confidence


def build_classifier(num_classes: int) -> Sequential:
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                     activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Head
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # 1) Load data
    df = load_dataframe(DATA_DIR, CSV_NAME)

    # 2) Train/test split (stratified)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df['PatchLabel']
    )

    train_df.to_csv('train_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)

    # 3) Encode labels using only train then transform test
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['PatchLabel'].values)
    test_labels = label_encoder.transform(test_df['PatchLabel'].values)

    num_classes = len(label_encoder.classes_)
    y_train = to_categorical(train_labels, num_classes=num_classes)
    y_test = to_categorical(test_labels, num_classes=num_classes)

    # 4) Load images
    X_train = load_images(train_df['Patch'].tolist())
    X_test = load_images(test_df['Patch'].tolist())

    # 5) Build model
    model = build_classifier(num_classes=num_classes)

    # 6) Train with early stopping and checkpoint
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks
    )

    # 7) Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    # 8) Save model (best already saved), and label classes for inference
    model.save(MODEL_PATH)
    np.save(CLASSES_PATH, label_encoder.classes_)

    # 9) Example prediction on a random train sample
    i = random.choice(train_df.index)
    sample_path = train_df.loc[i, 'Patch']
    sample_img = read_image_safe(sample_path)
    if sample_img is not None:
        # Make prediction using helper function
        pred_label, confidence = predict_image(model, sample_img, label_encoder)
        
        # Get true label for comparison
        true_label = train_df.loc[i, 'PatchLabel']
        
        print(f"Sample: {sample_path}")
        print(f"True label: {true_label} | Predicted: {pred_label}")
        print(f"Confidence: {confidence:.4f}")
        
        # Display image (convert RGB back to BGR for cv2.imshow)
        img_display = cv2.cvtColor((sample_img * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Predicted: {pred_label} | True: {true_label}", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Could not read sample image at: {sample_path}")

if __name__ == '__main__':
    main()


