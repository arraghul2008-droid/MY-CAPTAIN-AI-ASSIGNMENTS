import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


image_path = 'input_image.jpg' # REPLACE with your actual image filename
print(f"Loading image from {image_path}...")


img = cv2.imread(image_path)

if img is None:
    print("Image not found! Generating a synthetic satellite image for demonstration...")
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img[0:100, 0:100] = [255, 0, 0] 
    img[0:100, 100:200] = [0, 255, 0] 
    img[100:200, :] = [128, 128, 128] 

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Image Shape: {img.shape}") # (Height, Width, Bands)

new_shape = (img.shape[0] * img.shape[1], img.shape[2])
X = img.reshape(new_shape)

print(f"Data reshaped for Random Forest: {X.shape}")


y = np.zeros(X.shape[0])

gray_vals = np.mean(X, axis=1)
y[gray_vals > 100] = 1 
y[gray_vals <= 100] = 0

print("Labels generated.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Model (this may take a moment)...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Training Complete.")

print("Classifying the full image...")

full_prediction = rf.predict(X)

classified_map = full_prediction.reshape(img.shape[0], img.shape[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Satellite Image")
axes[0].axis('off')

axes[1].imshow(classified_map, cmap='jet')
axes[1].set_title("Classified Map (Random Forest)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("Project Executed Successfully.")