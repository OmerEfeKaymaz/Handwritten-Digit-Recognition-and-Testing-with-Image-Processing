import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))

# Calculate moments for each image
moments_list = []
for image in digits.images:
    img = np.array(image, dtype=np.uint8)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_moments = []
    for cnt in contours:
        moments = cv2.moments(cnt)
        digits_moments.extend(list(moments.values()))
    moments_list.append (digits_moments)    
       
max_moments_length = max(len(row) for row in moments_list)
moments_list_padded = [row + [0]* (max_moments_length - len(row)) for row in moments_list]
           
moments_array = np.array(moments_list_padded)

# Merge moments with original data
merged_df = pd.concat([pd.DataFrame(moments_array), pd.DataFrame(data)], axis=1)
df_data = merged_df.astype('float32')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_data, digits.target, test_size=0.2, shuffle=False)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(df_data.shape[1],)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)