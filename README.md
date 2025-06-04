# 📈 CreditInsight: Predicting Loan Default Risk Using ML

## 📊 Project Overview

Credit risk prediction plays a vital role in helping financial institutions minimize losses and make informed lending decisions. This project builds machine learning models using historical applicant data to predict whether a person is likely to default on a loan. The models classify applications into **Default** and **Non-Default** categories.

We implemented and evaluated four classification algorithms using standardized data and accuracy as the evaluation metric:
**Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**.

These models were chosen to provide a mix of interpretability, robustness, and predictive power:
- **Logistic Regression** is a linear model that estimates the probability of default and serves as a reliable, interpretable baseline.
- **Decision Tree** uses a series of feature-based splits to make decisions, offering transparency in how predictions are made.
- **Random Forest** builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.
- **SVM** works by finding the optimal boundary between classes and performs well in high-dimensional, scaled feature spaces.

By evaluating a variety of models, we aimed to compare their strengths and identify the most effective approach for credit risk prediction using accuracy as the main performance metric.


## 💻 Technologies Used

- Python 3
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn


## 🗃️ Dataset Overview

The dataset contains **1,000 records** and **9 features** related to applicant demographics and financial health.

| Column Name  | Type       | Description                                |
|--------------|------------|--------------------------------------------|
| age          | Integer    | Age of the applicant                       |
| ed           | Categorical| Education level                            |
| employ       | Integer    | Years of employment                        |
| address      | Integer    | Years at current address                   |
| income       | Float      | Annual income (in $1,000s)                 |
| debtinc      | Float      | Debt-to-income ratio (%)                   |
| creddebt     | Float      | Credit card debt                           |
| othdebt      | Float      | Other debt                                 |
| default      | Binary     | Target label: 1 = Default, 0 = Non-Default |


### 🔧 Preprocessing Steps

- **Removed missing values** to ensure the dataset was clean and complete.
- **Separated the features and target variable**, with the target being whether a customer defaulted or not.
- **Standardized the numerical features** using `StandardScaler` so that all values were on the same scale, which helps models like Logistic Regression and SVM perform better.
- Split the dataset into **training and test sets** to evaluate model performance on unseen data.


## 📊 Results Summary

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 82%      |
| Decision Tree       | 80%      |
| Random Forest       | 85%      |
| SVM (Optimized)     | 83%      |

> 📌 Accuracy was measured using `.score()` on the test set. Other metrics like precision, recall, and F1-score were not calculated in this project.


### 🔍 Key Takeaways

- **Debt-to-Income Ratio (`debtinc`) and Credit Card Debt (`creddebt`) stood out as important features**  
  Applicants with a high debt-to-income ratio or large credit card balances were more likely to default. These financial indicators directly reflect an individual’s ability to manage additional debt, making them strong predictors of credit risk.

- **Longer employment (`employ`) and stable residence (`address`) were associated with lower risk**  
  Individuals who have been employed or living at the same address for many years showed lower default rates. This suggests financial and lifestyle stability, which lenders often interpret as a sign of lower credit risk.

- **Standardizing the data was crucial for certain models**  
  Algorithms like Logistic Regression and SVM performed significantly better after scaling the features. Without standardization, models can become biased toward features with larger numerical values, leading to reduced accuracy.

- **Random Forest was the most accurate model overall**  
  Among all models tested, Random Forest achieved the highest accuracy. This suggests that combining multiple decision trees (an ensemble method) provides a more reliable prediction by reducing overfitting and improving generalization.


## 🚀 How to Run the Project

1. Clone the repository
2. Open the `CreditRiskPrediction.ipynb` notebook
3. Run all cells to see model training, predictions, and accuracy scores


## ✅ Conclusion

This project demonstrates how machine learning can be used to assess credit risk based on customer data. By testing multiple models and comparing their performance, we found that Random Forest delivered the most accurate results, while Logistic Regression and SVM offered strong, interpretable alternatives when combined with proper preprocessing. These insights highlight the importance of both model selection and data preparation in building effective predictive systems.

While the current implementation uses accuracy as the evaluation metric, future work could involve exploring precision, recall, and F1-score for a more balanced assessment, especially in cases where false positives or false negatives carry a high cost. Additionally, incorporating more real-world features like payment history, credit utilization, and loan type could help improve model performance and relevance.

This project provides a solid foundation for further exploration in financial analytics, and showcases the value of combining statistical understanding with practical machine learning techniques.
