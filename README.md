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

### ------------------------------------------------------------------------------------------ Issues -------------------------------------------------------------------------------------------------------------------
### 1.  Issues Faced While Incorporating ResNet into My Pretrained CNN and Precautions to Be Taken

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

### 2.  Issues Faced While Incorporating load, preprocess, and split image data for training a binary classification model (non-cancer vs. cancer)

### Summary of What Was Done:

You were working on a machine learning project where you wanted to load, preprocess, and split image data for training a binary classification model (non-cancer vs. cancer). Here's the sequence of actions you were taking:

1. **Loading and Preprocessing Images**:
   - You wrote a function (`load_and_preprocess_image`) to load images from file paths, resize them to 224x224 pixels, convert them to NumPy arrays, normalize them to the [0, 1] range, and add a batch dimension.
   
2. **Creating Datasets**:
   - You used the function to load non-cancer and cancer images from their respective directories, processed them, and then concatenated the images into one array (`X`).
   
3. **Creating Labels**:
   - You created labels for the images: `0` for non-cancer images and `1` for cancer images.

4. **Splitting the Data**:
   - You used `train_test_split` from Scikit-Learn to split the data into training and validation sets for model training.

### Error Encountered:
The error you encountered was:

```plaintext
ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 5 dimension(s) and the array at index 1 has 1 dimension(s)
```

This error occurred when you tried to concatenate the non-cancer and cancer images into a single dataset using `np.concatenate`. The issue was that the images were being loaded with an extra batch dimension (i.e., they had a shape of `(1, 224, 224, 3)` for each image), causing a mismatch when trying to concatenate them together.

### How the Issue Was Solved:

1. **Understanding the Problem**:
   - The `load_and_preprocess_image` function added an extra batch dimension to each image (with shape `(1, 224, 224, 3)`), causing the resulting arrays to be 5D instead of the expected 4D. When you tried to concatenate them, NumPy couldn't handle the dimension mismatch.

2. **Fixing the Solution**:
   - Instead of directly using `np.array()` to wrap the preprocessed images in `X_non_cancer` and `X_cancer`, you modified the code to first create a list of processed images and then used `np.concatenate` to merge them along the batch dimension (axis=0).
   - This resulted in a 4D array with the shape `(num_images, 224, 224, 3)` for both non-cancer and cancer images, which was the expected input shape for training the model.

3. **Code Modification**:
   ```python
   # Load and preprocess images into lists
   X_non_cancer = [load_and_preprocess_image(path) for path in image_paths_non_cancer if path is not None]
   X_cancer = [load_and_preprocess_image(path) for path in image_paths_cancer if path is not None]

   # Concatenate images properly into 4D arrays
   X_non_cancer = np.concatenate(X_non_cancer, axis=0)
   X_cancer = np.concatenate(X_cancer, axis=0)

   # Combine data and create labels
   X = np.concatenate([X_non_cancer, X_cancer], axis=0)
   y = np.concatenate([np.zeros(len(X_non_cancer)), np.ones(len(X_cancer))], axis=0)

   # Split data into training and validation sets
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

### Conclusion:
You successfully solved the dimensionality mismatch by ensuring that the images were loaded and concatenated correctly as 4D arrays, making them suitable for training the classification model. This fix allowed you to proceed with creating the training and validation sets and eventually move forward with model training.



### -------------------------------------------------------------------------------- Defınatıons-------------------------------------------------------------------------------------------

The **F1 score** is a performance metric for classification tasks that combines precision and recall into a single value. It is particularly useful when dealing with imbalanced datasets, where one class significantly outnumbers the other(s).

---

### **Formula**
The F1 score is defined as the harmonic mean of precision and recall:

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Where:
- **Precision** (Positive Predictive Value):
  \[
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  \]
  Precision measures how many of the predicted positive results are actually positive.

- **Recall** (Sensitivity or True Positive Rate):
  \[
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  \]
  Recall measures how many of the actual positive results are correctly predicted.

---

### **Why Use the F1 Score?**
- **Balances Precision and Recall**: Useful when you want to balance both metrics, especially when improving one might harm the other.
- **Focuses on Positive Predictions**: Helps evaluate models in scenarios where the positive class is more important (e.g., detecting diseases, fraud detection).

---

### **Interpretation**
- \( F1 = 1 \): Perfect precision and recall.
- \( F1 = 0 \): Either precision or recall is zero (complete failure in prediction).

---

### **When to Use**
- When the dataset is imbalanced.
- When false positives and false negatives carry different costs.
- When optimizing both precision and recall is critical.

---

### **Example**
Consider a binary classification problem where:
- True Positives (\( TP \)) = 80
- False Positives (\( FP \)) = 20
- False Negatives (\( FN \)) = 10

1. **Precision**:
   \[
   \text{Precision} = \frac{80}{80 + 20} = 0.8
   \]

2. **Recall**:
   \[
   \text{Recall} = \frac{80}{80 + 10} = 0.888
   \]

3. **F1 Score**:
   \[
   F1 = 2 \cdot \frac{0.8 \cdot 0.888}{0.8 + 0.888} = 0.842
   \]

The F1 score of 0.842 indicates a good balance between precision and recall.

### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### References
### Resnet lınk 
https://github.com/akshatapatel/Breast-Cancer-Image-Classification/blob/master/ResNet.ipynb

Output:
| **Model**                | **Validation Score** | **Score Behavior**                                                                                   | **Observation**                                                                                                                                                                  |
|--------------------------|-----------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **With Skip Connections** | 0.84                 | Improves consistently during training                                                              | Learns better due to the inclusion of skip connections, which help mitigate vanishing gradient issues and improve feature learning.                                              |
| **Without Skip Connections** | 0.49                 | Remains almost constant (between 0.48 - 0.50) across all epochs                                     | A deep model with 25 layers (20 convolutional and 5 max-
pooling) struggles to learn on the dataset without skip connections, leading to poor performance and stagnant validation scores. |

### Resnet Applıed on my mamogram dataset

Best Epoch (Val Accuracy): 0
accuracy        0.948941
loss            0.203660
val_accuracy    0.952769
val_loss        0.215294
Name: 0, dtype: float64
Best Epoch (Val Loss): 0
accuracy        0.948941
loss            0.203660
val_accuracy    0.952769
val_loss        0.215294





### Other Work On Ultrasound Images 

link1: https://www.kaggle.com/code/aditimondal23/vgg19-breast
Developed a custom image classifier using a pre-trained VGG19 model for a 3-class classification task (e.g., benign, normal, malignant). The model was fine-tuned by adding fully connected layers with dropout, batch normalization, and L2 regularization to enhance performance and prevent overfitting. Training incorporated early stopping (based on validation loss) and model checkpointing (saving the best model by validation accuracy). Performance was evaluated using metrics such as accuracy, loss, and ROC curves. Visualized predictions with confidence scores on test images for better interpretability.
VGG19 Model Creation
The VGG19 model is used as a base (pre-trained on ImageNet), with additional layers added to adapt it to the classification task. Key steps:

Freeze Base Model Layers: Prevents training on pre-trained VGG19 layers to retain learned features.
Add Custom Layers:
Flatten Layer: Converts the feature map to a vector for fully connected layers.
BatchNormalization: Normalizes intermediate layer outputs to stabilize training.
Dense Layers: Fully connected layers with L2 regularization to prevent overfitting.
Dropout: Randomly deactivates neurons during training for regularization.
Output Layer: A softmax layer with 3 output units, representing 3 classes.
The model is compiled with:

Optimizer: Stochastic Gradient Descent (SGD) with momentum.
Loss Function: Categorical crossentropy for multi-class classification.
Metrics: Tracks accuracy during training.

--- Model Evaluation Metrics ---
Train accuracy: 0.9950
Validation accuracy: 0.8507
Test accuracy: 0.8034
F1 Score: 0.8009
Kappa Score: 0.6748
ROC AUC Score: 0.9266
Precision: 0.8062
Recall: 0.8034

link2: https://www.kaggle.com/code/bevenky/gemini-1-5-for-ultrasound-tumor-analysis

The results from the **Gemini 1.5 API** for breast ultrasound images are providing detailed analysis based on the **features indicative of malignancy**. Here’s a breakdown of the key findings and actions based on the AI’s analysis:

### **Key Findings:**
1. **Irregular Borders**:
   - Irregular, ill-defined borders are a significant indicator of malignancy. Malignant tumors often present with this feature, unlike benign lesions that tend to have well-defined, smooth borders.

2. **Hypoechoic Regions**:
   - Hypoechoic (darker) areas in the image suggest areas of low-density tissue, which could indicate a mass or tumor. This is typically seen in malignant tumors.
   
3. **Heterogeneous Echotexture**:
   - Malignant tumors often display varying levels of echogenicity within the mass, leading to a heterogeneous appearance. This differs from benign lesions that are generally homogeneous.

4. **Spiculated Masses**:
   - Some images show spiculated (spiky) masses, a hallmark feature of malignant tumors, which may invade surrounding tissues.

5. **Potential for Tumors**:
   - The image suggests the presence of a tumor but does not provide enough detail for a definitive diagnosis. The features observed, such as irregular borders and hypoechoic areas, are concerning for malignancy.

6. **Lack of Acoustic Shadowing and Calcifications**:
   - The absence of posterior acoustic shadowing and calcifications makes it difficult to definitively classify the mass as malignant in some cases. These features are often indicative of tumors but may not always be present.

---

### **Actionable Insights from the AI Results:**
- **Further Investigation Needed**:
  - The AI suggests that while the images show concerning features, additional imaging (e.g., mammography, MRI) and clinical data are necessary for a more accurate diagnosis.
  - A **biopsy** is recommended to confirm the nature of the mass.

- **Recommendation for Healthcare Professionals**:
  - The AI results indicate that **consultation with a radiologist** is needed for a comprehensive analysis. A radiologist can review the full sequence of images and consider other factors like patient history.
  
- **Clinical Follow-up**:
  - The AI points out that features such as **irregular borders**, **hypoechoic regions**, and **heterogeneous echotexture** are common in malignancies but may also occur in other conditions. Further clinical evaluation, including biopsies or additional scans, is needed to establish a definitive diagnosis.

---

### lınk3 https://www.kaggle.com/code/datnguyen1235/breast-cancer-classification-accuracy-100

### **Summary of the Specific Results for Malignant Images**:

| **Image**                        | **Findings**                                                                                       | **Recommendation**                                                                                             |
|-----------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **malignant (32).png**            | Large hypoechoic mass with irregular borders and spiculated shape. Suggestive of malignancy.      | **Biopsy** needed to confirm malignancy. Further investigation required.                                       |
| **malignant (175).png**           | Hypoechoic region with irregular borders. Challenging without video context. Possible malignancy. | **Further analysis** required. Clinical consultation recommended.                                            |
| **malignant (78).png**            | Irregular borders, heterogeneous echotexture, and hypoechoic regions. Suggestive of malignancy.    | **Biopsy** to confirm tumor nature. Further investigation is necessary.                                        |
| **malignant (193).png**           | Hypoechoic, round-shaped mass with irregular borders. Heterogeneous echotexture. Suspicious mass.  | **Further imaging** and biopsy required to confirm malignancy.                                                 |
| **malignant (36).png**            | Hypoechoic region with irregular borders. Suggests possible malignancy.                           | Requires **additional imaging** and clinical evaluation.         |


Here’s the data you provided formatted into a **table** for better readability:

| **Epoch** | **Train Loss** | **Learning Rate** | **Val Loss** | **Val Accuracy** | **Val Recall** | **Val F1-score** | **Training Time per Epoch** |
|-----------|----------------|-------------------|--------------|------------------|----------------|------------------|----------------------------|
| 1         | 0.3645         | 0.00098284         | 0.3645       | 95.73%           | 95.50%         | 95.73%           | 28s                        |
| 2         | 0.1120         | 0.00096479         | 0.0086       | 99.15%           | 99.43%         | 99.26%           | 28s                        |
| 3         | 0.0796         | 0.00094670         | 0.0388       | 99.15%           | 99.10%         | 99.26%           | 27s                        |
| 4         | 0.1749         | 0.00092857         | 0.0245       | 99.15%           | 99.10%         | 99.26%           | 27s                        |
| 5         | 0.1526         | 0.00091040         | 0.2311       | 98.29%           | 98.20%         | 98.51%           | 27s                        |
| 6         | 0.0867         | 0.00089219         | 0.2569       | 79.49%           | 64.87%         | 60.03%           | 27s                        |
| 7         | 0.1121         | 0.00087394         | 0.0872       | 99.15%           | 99.10%         | 99.26%           | 28s                        |
| 8         | 0.0586         | 0.00085565         | 0.0261       | 99.15%           | 99.10%         | 99.26%           | 27s                        |
| 9         | 0.0374         | 0.00083731         | 0.0048       | 100%             | 100%           | 100%             | 28s                        |

### Key Points:
- **Epoch 9** shows the highest performance with **100% accuracy**, **100% recall**, and **100% F1-score** on the validation set.
- There are **warnings related to the `os.fork()`** call during the training process. This may indicate compatibility issues when using multiprocessing with certain libraries like JAX.
- The **learning rate** gradually decreases, and the **loss** continues to improve as the epochs progress, indicating successful training.

---

### **Training Performance Analysis**:
- The model is steadily improving as seen in the **decreasing loss** and **increasing F1-score**.
- **Epoch 6** shows a significant drop in performance (F1-score of **60%**), likely due to the **model overfitting** or **a problematic learning rate** for that specific epoch.
- From **Epoch 7 onwards**, performance improves again with high **accuracy** and **F1-score**.

---

### lınk4 https://www.kaggle.com/code/aishibiswas1/googlenet-trial1

This code trains a **breast cancer classification model** using ultrasound images, categorizing them into **Benign**, **Malignant**, and **Normal** classes. It utilizes **InceptionV3** with pre-trained ImageNet weights for transfer learning. The model includes a **GlobalAveragePooling2D** layer followed by a **Dense** layer for classification.

The dataset is processed using **ImageDataGenerator**, with 20% of the data reserved for validation. The model is trained for 10 epochs with **categorical cross-entropy loss** and **Adam optimizer**. After training, the model's performance is evaluated using **accuracy**, **precision**, **recall**, and **F1-score** metrics.

**Visualizations** include a bar chart for precision, recall, and F1-score, a **confusion matrix**, and **ROC** and **Precision-Recall curves** for each class. Predictions on the test set are also shown, including true and predicted labels for a few test images.

This workflow allows you to train, evaluate, and visualize the performance of a deep learning model on breast ultrasound images.

precision    recall  f1-score   support

      Benign       0.57      0.69      0.62       178
   Malignant       0.31      0.23      0.26        84
      Normal       0.15      0.11      0.13        53

    accuracy                           0.47       315
   macro avg       0.34      0.34      0.34       315
weighted avg       0.43      0.47      0.44       315


### Link5 https://github.com/nyukat/breast_cancer_classifier

## Summary

This repository provides an implementation of a deep learning model for breast cancer classification in mammograms. The model achieves high accuracy in predicting malignant and benign findings, potentially improving radiologists' performance in breast cancer screening.

Here's a breakdown of the key points:

* **Model Type:** Deep convolutional neural network
* **Inputs:** Four mammogram images (one for each standard view: L-CC, R-CC, L-MLO, R-MLO)
* **Outputs:** Probability of benign and malignant findings for each breast (left and right)
* **Performance:**
    * Achieves an AUC (Area Under the Curve) of 0.895 for identifying malignant cases.
    * Accuracy is slightly lower for benign cases.
    * A hybrid model combining radiologist predictions with the model's predictions shows even better accuracy than either method alone.

## Disclaimer

It's important to note that the code provided in this repository is outdated (created in 2019). The authors recommend contacting them for information on their latest and most accurate models. Here's the contact information from the repository:

* Email: krzysztof.geras@nyulangone.org

## Overall Performance

The deep learning model described here demonstrates promising results in breast cancer classification. It achieves good accuracy in identifying malignant cases and can potentially be a valuable tool to assist radiologists in screening. However, it's crucial to use the most recent and well-performing models for clinical applications.









