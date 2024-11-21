# Breast Cancer Prediction

First, I have two files. One is `datacombined`, which tells us which patients have cancer and which do not. The "Cancer" sheet (105 rows × 561 columns) represents cancer, and the "Non-Cancer" sheet (4998 rows × 561 columns) represents non-cancer.

I also have folders named "Cancer" and "Non-Cancer" in my OneDrive. The "Cancer" folder has two subfolders, and the "Non-Cancer" folder is divided into batches. The `datacombined` file and another Excel file called `data_validation_x` (322 rows × 16 columns) are used together. The "Non-Cancer" folder has a column called `InputFileName`, which is unique and tells us the location of DICOM images that have been converted to JPG format and stored in the respective folder.

For the "Cancer" folder, I needed to use two Excel files: `datacombined` (Sheet 1) and `data_validation_x`. What I did was use the common columns between the two files to create a unique key.

### Create a new column 'UniqueKey' by concatenating the specified columns:

```python
datacombined_df_1['UniqueKey'] = datacombined_df_1[['InputFileName', 'StudyDate', 'StudyTime', 'PatientID.x']].astype(str).agg('-'.join, axis=1)
```

### Output will look like:

```
FILE0004-20141225-112603-DIT439120108
```

Out of the 105 rows in `datacombined`, I was able to match 87 rows (× 577 columns). The columns increased due to merging the two Excel files. `datacombined` was the file provided by the hospital, which represented the validation of the images.

---

I have saved the file to navigate images as `CANCER.csv` that has 87 matches and ``NON CANCER.csv` 

### Retrieving images using Cancer and non cancer fıle that we have created.
```python
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the CSV files containing image metadata
matched_df = pd.read_csv('NON CANCER.csv', sheet_name=0)  # First sheet (Non-Cancer Data)
merged_cancer = pd.read_csv('CANCER.csv', sheet_name=1)  # Second sheet (Cancer Data)

# Define the root folder paths where the images are stored
root_folder_non_cancer = r'C:\Users\kenza.chenni\Desktop\acıbademsana\non cancer'
root_folder_cancer = r'C:\Users\kenza.chenni\Desktop\acıbademsana\cancer'

# Get the image filenames from the metadata DataFrames (Non-Cancer and Cancer)
image_names_non_cancer = matched_df['InputFileName'].astype(str).tolist()
image_paths_cancer = merged_cancer['AbsolutePath'].astype(str).tolist()

# Function to find the image path from subfolders in a given folder
def get_image_path_from_subfolders(image_name, root_folder):
    for root, dirs, files in os.walk(root_folder):
        if image_name in files:
            return os.path.join(root, image_name)  # Return the full path of the image
    return None  # Return None if image is not found

# Function to prepare the data by loading images and assigning labels
def prepare_data(image_names_non_cancer, image_paths_cancer, root_folder_non_cancer, root_folder_cancer):
    images = []
    labels = []
    
    # Process Non-Cancer images (label 0)
    for image_name in image_names_non_cancer:
        image_path = get_image_path_from_subfolders(image_name, root_folder_non_cancer)
        if image_path:
            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224 pixels
            img_array = image.img_to_array(img) / 255.0  # Normalize the pixel values to [0, 1]
            images.append(img_array)
            labels.append(0)  # Assign label 0 for Non-Cancer

    # Process Cancer images (label 1)
    for image_path in image_paths_cancer:
        image_name = os.path.basename(image_path)  # Extract the filename from the full path
        img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224 pixels
        img_array = image.img_to_array(img) / 255.0  # Normalize the pixel values to [0, 1]
        images.append(img_array)
        labels.append(1)  # Assign label 1 for Cancer

    return np.array(images), np.array(labels)

# Prepare the data for training the model
images, labels = prepare_data(image_names_non_cancer, image_paths_cancer, root_folder_non_cancer, root_folder_cancer)

# Optionally, split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Optionally, you can encode the labels using LabelEncoder if needed for classification
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Example of how to define a simple CNN model for image classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification (Cancer vs Non-Cancer)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_val, y_val_encoded))

# Load the images and labels
X, y = prepare_data(image_names_non_cancer, image_paths_cancer, root_folder_non_cancer, root_folder_cancer)

# Encode labels if necessary (if the labels are not already in numerical form)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# CNN Model
def create_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Softmax for binary classification (0 and 1)
    return model

# Create and compile the model
model = create_cnn_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predictions and evaluation metrics
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-cancer', 'Cancer'], yticklabels=['Non-cancer', 'Cancer'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred_classes))

```

### output 

```
Test Loss: 0.14547954499721527
Test Accuracy: 0.9929577708244324

```
# Classification Metrics

Here are the classification metrics for the model evaluation:

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.99      | 1.00   | 1.00     | 984     |
| 1             | 1.00      | 0.30   | 0.46     | 10      |
| **Accuracy**  |           |        | 0.99     | 994     |
| **Macro avg** | 1.00      | 0.65   | 0.73     | 994     |
| **Weighted avg** | 0.99   | 0.99   | 0.99     | 994     |


### Issues Faced While Incorporating ResNet into My Pretrained CNN and Precautions to Be Taken

When trying to extend my pretrained CNN model (`cnn_model_onImageandExcel.h5`) by integrating it with ResNet50 for further feature extraction, I faced the following challenges:

1. **Error in Extracting Feature Maps**:
   - I attempted to extract feature maps from the CNN model by accessing one of its convolutional layers (`cnn_model.layers[-3].output`), but I encountered an error:  
     `AttributeError: The layer sequential_1 has never been called and thus has no defined input.`  
   - **Cause**: This error occurred because the model was not properly initialized or compiled when I tried to access its layers. Specifically, the CNN model was loaded, but its layers had not been executed yet, so they didn't have defined inputs or outputs.
   
2. **Warning Regarding Model Compilation**:
   - Upon loading the model, I received the warning:  
     `Compiled the loaded model, but the compiled metrics have yet to be built.`  
   - **Cause**: The model was loaded without being compiled, meaning the metrics for evaluation (e.g., accuracy) were not yet available. This warning is typical when working with saved models that haven't been compiled or evaluated after loading.

### Precautions to Take When Attempting the Integration Again

1. **Ensure Proper Initialization**:
   - Before extracting layers or modifying the model, ensure that the model is properly initialized by compiling it, even if you're not training it again immediately.
   - Use `model.compile()` to make sure the model’s metrics and layers are properly set up before interacting with them.

2. **Extracting Intermediate Layers Correctly**:
   - To avoid errors, use a `Model` instance that includes the input layer and the desired intermediate output layer (e.g., a convolutional layer) from the pretrained CNN.  
   - Example:  
     ```python
     feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
     ```
   - This will extract features from a specific layer rather than trying to use the entire model's output.

3. **Integrating ResNet50 Properly**:
   - When adding ResNet50 to the CNN model, ensure that the output of the CNN (feature maps) is used as the input for ResNet50.  
   - This allows the CNN’s features to be further processed by ResNet50 for enhanced feature extraction.
   - Example:
     ```python
     resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=cnn_features)
     ```

4. **Model Compilation After Changes**:
   - After modifying the model (e.g., adding ResNet50), always compile the model again before training or evaluating.  
   - Use the following compilation step:
     ```python
     combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     ```

5. **Monitor Warnings**:
   - Always pay attention to warnings regarding model compilation, especially after loading a pretrained model.  
   - Ensure that any metrics and layers required for evaluation are properly set up by compiling the model before proceeding.

By following these precautions, I can safely integrate pretrained CNN models with additional networks like ResNet50 and avoid common pitfalls like uninitialized layers or missing compilation steps.

### Resnet lınk 
https://github.com/akshatapatel/Breast-Cancer-Image-Classification/blob/master/ResNet.ipynb

