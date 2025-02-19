# Sales Prediction Using Python

## Project Overview
This project focuses on predicting sales based on advertising budgets allocated to TV, Radio, and Newspaper using a linear regression model. The goal is to understand the impact of different advertising channels on sales and build a predictive model to estimate future sales.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
  
## Dataset
The dataset used in this project is `Advertising.csv`, which contains the following columns:
- **TV**: Advertising budget spent on TV (in thousands)
- **Radio**: Advertising budget spent on Radio
- **Newspaper**: Advertising budget spent on Newspaper
- **Sales**: Corresponding sales (in millions)

## Approach
1. **Data Exploration**: Analyze the dataset to understand the relationships between advertising channels and sales.
2. **Model Training**: Use a linear regression model to predict sales based on advertising budgets.
3. **Evaluation**: Evaluate the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

## Code Explanation
1. **Import Libraries**: The project uses libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
2. **Load Data**: The dataset is loaded and checked for missing values.
3. **Exploratory Data Analysis (EDA)**: Visualizations are created to explore the relationships between variables.
4. **Model Training**: The data is split into training and testing sets, and a linear regression model is trained.
5. **Evaluation**: The model's performance is evaluated using various metrics, and predictions are visualized against actual sales.

## Results
- **R² Score**: ~0.90, indicating a strong predictive power of the model.
- **Key Insight**: TV advertising has the highest impact on sales compared to Radio and Newspaper.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

## Run the Jupyter Notebook:
- Open `Sales_Prediction.ipynb` in Jupyter Notebook.
- Execute each cell to run the analysis and model training.
