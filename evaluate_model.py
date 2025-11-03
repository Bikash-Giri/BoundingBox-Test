import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path


random_image = "/Users/bikashgiri/Desktop/bikashprofile.png"

# ================================
# 1️⃣ Load model
# ================================
model_path = "my_model.h5"  # update path if needed
model = load_model(model_path)
print("✅ Model loaded successfully.")

# ================================
# 2️⃣ Load your test dataset
# ================================
# Example: your test CSV file or variable
# If you already have test_df loaded in memory, comment this line
test_df = pd.read_csv("test_split.csv")  # update if needed

# test_df should contain columns like: 'Patch' (image path) and 'Label'
X_test = []
y_test = []

print("📂 Loading test images...")
for _, row in test_df.iterrows():
    img_path = row['Patch']
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Warning: could not read image {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # must match training size
    X_test.append(img.astype('float32') / 255.0)
    y_test.append(row['PatchLabel'])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"✅ Loaded {len(X_test)} test images.")

# ================================
# 3️⃣ One-hot encode labels
# ================================
num_classes = model.output_shape[-1]
label_mapping = {0: 0, 5: 1, 6: 2, 7: 3}
y_test_mapped = np.array([label_mapping[y] for y in y_test])

num_classes = model.output_shape[-1]
y_test_cat = to_categorical(y_test_mapped, num_classes=num_classes)


# ================================
# 4️⃣ Evaluate model
# ================================
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# ================================
# 5️⃣ Predict and visualize samples
# ================================
pred = model.predict(X_test)

pred_classes = np.argmax(pred, axis=1)

# Show 5 random samples
num_show = 20
indices = np.random.choice(len(X_test), num_show, replace=False)

# for i in indices:
#     plt.imshow(X_test[i])
#     plt.title(f"True: {y_test[i]}, Pred: {pred_classes[i]}")
#     plt.axis('off')
#     plt.show()

# pred = model.predict(random_image)  # shape: (1, num_classes)
# pred_class = np.argmax(pred, axis=1)[0]


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# Example: single image path


# # check if file exists
# if not os.path.exists(random_image):
#     raise FileNotFoundError(f"File not found at {random_image}")

# # Convert to RGB if your model was trained on RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Resize to match training input size
# img = cv2.resize(img, (64, 64))  # or (64,64) depending on your model

# # Normalize
# img = img.astype('float32') / 255.0

# # Add batch dimension
# img_input = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

# pred = model.predict(img_input)  # shape: (1, num_classes)
# pred_class = np.argmax(pred, axis=1)[0]

# print(f"Predicted class index: {pred_class}")
# print(f"Predicted probabilities: {pred[0]}")


# # Class labels (optional)
# class_labels = ["0", "5", "6", "7"]
# # Define video path
# video_path = "/Users/bikashgiri/Downloads/wedding-highlights.mp4"

# # Open video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise FileNotFoundError(f"Could not open video: {video_path}")

# # Get video FPS (frames per second)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"FPS: {fps}")

# # Calculate frame interval for 5 seconds
# frame_interval = int(fps * 5)

# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # ✅ Process frame every 5 seconds
#     if frame_count % frame_interval == 0:
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (64, 64))  # adjust if needed
#         img = img.astype("float32") / 255.0
#         img_input = np.expand_dims(img, axis=0)

#         pred = model.predict(img_input, verbose=0)
#         print(pred)
#         pred_class = np.argmax(pred, axis=1)[0]

#         print(pred_class)
#         label = class_labels[pred_class] if pred_class < len(class_labels) else str(pred_class)

#         # Display on frame
#         cv2.putText(frame, f"Predicted: {label}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         print(f"Frame {frame_count}: Predicted {label}")

#     cv2.imshow("Video Prediction", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Clean up
# cap.release()
# cv2.destroyAllWindows()