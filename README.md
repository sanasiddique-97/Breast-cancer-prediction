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



