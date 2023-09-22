# Wine Quality Prediction and Clustering Analysis



![image](https://github.com/Theo-and-Adam/clustering_project/assets/118470135/91e36d30-5bb6-42ad-aa2b-12d155ecf749)


## Project Overview
The Wine Quality Prediction and Clustering Analysis project is a data science consultancy endeavor commissioned by the CodeUP Institute. The objective of this project is to predict the quality of red and white wine while incorporating unsupervised learning techniques, particularly clustering. We will work with a comprehensive dataset containing information about different wines and their quality ratings. Our primary focus will be on using clustering to gain insights into the data and exploring how clustering can impact the performance of machine learning models in predicting wine quality.


## Project Deliverables
### GitHub Repository:
- **README.md**: This document, including project description, data dictionary, project plan, initial questions, conclusion, and instructions on reproducing the work.
- **final_notebook.ipynb**: A Jupyter notebook containing the entire project pipeline, complete with documentation and code comments.
- **Python modules**: `acquire.py` (or `wrangle.py` and `explore.py`) for data wrangling and preprocessing.


## Project Dictionary
| Variable             | Description                                      | Data Type       |
| -------------------- | ------------------------------------------------ | --------------- |
| fixed_acidity        | The fixed acidity of the wine.                   | Numeric (float) |
| volatile_acidity     | The volatile acidity of the wine.                | Numeric (float) |
| citric_acid          | The citric acid content in the wine.             | Numeric (float) |
| residual_sugar       | The residual sugar content in the wine.          | Numeric (float) |
| chlorides            | The chloride content in the wine.                | Numeric (float) |
| free_sulfur_dioxide  | The amount of free sulfur dioxide in the wine.   | Numeric (float) |
| total_sulfur_dioxide | The total sulfur dioxide content in the wine.    | Numeric (float) |
| density              | The density of the wine.                         | Numeric (float) |
| pH                   | The pH level of the wine.                        | Numeric (float) |
| sulphates            | The amount of sulphates in the wine.             | Numeric (float) |
| alcohol              | The alcohol content in the wine.                 | Numeric (float) |
| quality              | The quality rating of the wine (target variable).| Integer         |
| wine_type            | The type of wine (e.g., red or white).           | Categorical     |



## Project Plan
### Acquisition:
1. Collect and load the wine quality dataset.
2. Combine the red and white wine datasets.
3. Concatenate the datasets into one.

### Preparation:
1. Perform data preprocessing tasks, including adjustments to data types, handling missing values, and addressing outliers.
2. Prepare the data for exploration and modeling.

### Exploration:
1. Explore the data through visualization and statistical testing of hypothesis.
2. Utilize clustering techniques to gain insights into the data.
3. Evaluate the usefulness of clustering for the prediction task.
4. Draw conclusions from the exploration.

### Modeling:
1. Implement five different machine learning models; Linear Regression, Random Forest Regressor, Tweedie Regressor with Polynomial Features, Random Forest Regressor with Polynomial Features, and XGBRegressor with Polynomial Features
2. Compare the performance of these models.


### Presentation:
1. Developed a slide deck for the presentation.
2. Include an introduction, executive summary, findings, conclusion, and recommendation in the presentation.
3. 5-minute time limit.

### Steps to Reproduce
Acquire wine quality data from Data.World.
Merge red and white wine CSVs, adding a color identifier.
Clone repository in the local
Execute notebook with adjust hyperparameters


## Conclusion
The Wine Quality Prediction and Clustering Analysis project aims to predict wine quality while utilizing unsupervised learning techniques, with a focus on clustering for data exploration. By adhering to a structured plan and implementing best practices, we have analyzed the finding to proven that the increase in alcohol and free Sulfure Dioxide while decreasing the Chlorides will increase the wine quality.
