import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import cv2
import os
import glob

flower_categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model_path = 'D:/ComputerVision/flowers3.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")
model = tf.keras.models.load_model(model_path)

img_dir = "D:/ComputerVision/TestImage/Rose"
data_path = os.path.join(img_dir, '*')
files = glob.glob(data_path)

if not files:
    raise FileNotFoundError("Không tìm thấy ảnh nào trong thư mục")

num = 0 
for f1 in files:
    num = num + 1
    img = cv2.imread(f1)
    img = cv2.resize(img, (700, 700))  
    cv2.imshow('img', img)
    key = cv2.waitKey(0)

    if key == 13:
        break
cv2.destroyAllWindows()

chosen_image_path = f1
print("Ảnh được chọn:", chosen_image_path)

img_size = (224, 224)
test_image = load_img(chosen_image_path, target_size=img_size)
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) / 255.0  

result = model.predict(test_image)[0]
predicted_index = np.argmax(result)
predicted_label = flower_categories[predicted_index]
predicted_confidence = result[predicted_index] * 100

print(f"Dự đoán: {predicted_label} ({predicted_confidence:.2f}%)")
print("Chi tiết tỷ lệ dự đoán:")
for i, flower in enumerate(flower_categories):
    print(f"{flower}: {result[i] * 100:.2f}%")

img_final = cv2.imread(chosen_image_path)
img_final = cv2.resize(img_final, (700, 700))
cv2.putText(img_final, f"Prediction: {predicted_label} ({predicted_confidence:.2f}%)", 
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Result", img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('D:/ComputerVision/Result/result1.jpg', img_final)
