# Simple Salary Prediction Model (using Linear Regression)

## Project Overview

This repository contains a Google Colab notebook (`Simple Linear Regression.ipynb`) demonstrating how to build a simple linear regression model. The model predicts an individual's salary based on their years of experience using the `Salary_Data.csv` dataset.

The primary goal is to train a model on a portion of the dataset and then evaluate its ability to predict salaries on unseen data.

## Dataset

The dataset used is `Salary_Data.csv`. It consists of two columns:

1.  **YearsExperience**: The number of years of professional experience an individual has.
2.  **Salary**: The corresponding salary for that individual.

## Model

A **Simple Linear Regression** model is implemented. This model attempts to find a linear relationship between the independent variable (Years of Experience) and the dependent variable (Salary). The model learns this relationship from the training data and then uses it to make predictions.

The formula for simple linear regression is:
$$ \text{Salary} = \beta_0 + \beta_1 \times \text{YearsExperience} $$
Where:
- \( \beta_0 \) is the intercept (the predicted salary when years of experience is 0).
- \( \beta_1 \) is the slope (the change in salary for each one-year increase in experience).

## Files in this Repository

*   **`Simple Linear Regression.ipynb`**: The Google Colab notebook containing the Python code for data loading, preprocessing, model training, prediction, and visualization.
*   **`Salary_Data.csv`**: The dataset containing years of experience and corresponding salaries.
*   **`README.md`**: This file.

## How to Use/Run (Google Colab)

1.  **Open in Google Colab:**
    *   The easiest way is to click the "Open in Colab" badge if available at the top of the repository (you can generate one for your repo).
    *   Alternatively, go to [Google Colab](https://colab.research.google.com/).
    *   Select `File -> Upload notebook...` and upload the `Simple Linear Regression.ipynb` file.
    *   Or, select `File -> Open notebook...`, choose the "GitHub" tab, paste the URL of this repository, and select `Simple Linear Regression.ipynb`.

2.  **Upload the Dataset:**
    *   Once the notebook is open in Colab, you'll need to upload the `Salary_Data.csv` file to your Colab environment.
    *   In the left-hand sidebar, click on the "Files" icon (folder icon).
    *   Click the "Upload to session storage" button (upward arrow icon) and select `Salary_Data.csv` from your local machine.
    *   *Note: Files uploaded this way are temporary and will be deleted when the Colab runtime is recycled.*

3.  **Run the Cells:**
    *   Execute the cells in the notebook sequentially by clicking the "Play" button to the left of each cell or by selecting `Runtime -> Run all` from the menu.

4.  The notebook will:
    *   Load the `Salary_Data.csv` dataset.
    *   Split the data into training and testing sets.
    *   Train a linear regression model on the training set.
    *   Make predictions on the test set.
    *   Display two plots:
        *   Salary vs. Experience for the Training set, along with the regression line.
        *   Salary vs. Experience for the Test set, along with the regression line learned from the training set.

## Code Overview

The Python script within the notebook performs the following steps:

1.  **Import Libraries**:
    *   `numpy` for numerical operations.
    *   `pandas` for data manipulation and CSV file I/O.
    *   `matplotlib.pyplot` for plotting graphs.
    *   `sklearn.model_selection.train_test_split` for splitting the dataset.
    *   `sklearn.linear_model.LinearRegression` for implementing the regression model.

2.  **Load Data**:
    *   Reads `Salary_Data.csv` into a pandas DataFrame.
    *   Separates the features (Years of Experience, `X`) from the target variable (Salary, `Y`).

3.  **Split Data**:
    *   Divides the dataset into a training set (80%) and a test set (20%) using `train_test_split`. A `random_state` is set for reproducibility.

4.  **Train Model**:
    *   Initializes a `LinearRegression` model.
    *   Fits the model to the training data (`X_train`, `Y_train`).

5.  **Make Predictions**:
    *   Uses the trained model to predict salaries (`Y_pred`) for the test set features (`X_test`).

6.  **Visualize Results**:
    *   **Training Set Visualization**: Creates a scatter plot of the actual training data points (`X_train`, `Y_train`) and overlays the regression line predicted by the model for the training data.
    *   **Test Set Visualization**: Creates a scatter plot of the actual test data points (`X_test`, `Y_test`) and overlays the same regression line (learned from the training data) to see how well it generalizes to new data.

## Libraries Used

The necessary libraries are typically pre-installed in Google Colab environments. If you encounter any import errors, you can install them within a Colab cell using:
```python
!pip install numpy pandas matplotlib scikit-learn
```
*   NumPy
*   Pandas
*   Matplotlib
*   Scikit-learn

---
