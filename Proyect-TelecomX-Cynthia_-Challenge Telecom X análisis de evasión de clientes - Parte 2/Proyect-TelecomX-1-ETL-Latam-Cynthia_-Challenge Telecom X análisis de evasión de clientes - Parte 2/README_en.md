# 🎯 Machine Learning Models: Customer Churn Prediction

This project uses machine learning techniques to predict the likelihood of a customer churning. The analysis is based on a dataset with demographic and behavioral characteristics, applying models such as XGBoost, LightGBM, Random Forest, Logistic Regression, and an ensemble stacking model.

The objective is to develop models capable of early identification of customers at greatest risk of canceling their services. This will allow the company to implement preventive actions to improve retention, relying on a robust modeling pipeline from this initial phase. Patterns in contractual variables, use of digital services, and payment methods are studied, integrating exploratory analysis, correlation matrices, and explanations using SHAP.

> 🔄 **This repository corresponds to the second stage of the project**, focused on *Machine Learning*.
> The first part, dedicated to **data analysis and engineering** on the same dataset, can be found here: [Challenge2-Alura-Store](https://github.com/BarbaraAngelesOrtiz/Challenge2-Alura-Store).

---

## ✏️Main Goals

- Perform thorough data processing, including cleaning, encoding, and scaling.
- Analyze the relationship between variables to select the most relevant ones for prediction.
- Train and compare different classification models to detect churn.
- Evaluate the models with metrics that reflect their actual performance.
- Analyze the importance of the variables and their impact on prediction.
- Present conclusions with recommendations based on the key factors driving churn.

---

## 💡Activities Developed

- Process and prepare data to feed machine learning models.
- Build, train, and validate predictive models.
- Interpretation of results to extract valuable insights.
- Preparation of technical communications geared toward strategic decision-making.

---

## 📁 Project Structure

```bash
├── TelecomX_LATAM2_en.ipynb # English code
├── imag # Graphics
├── src # Dataset
├── requirements.txt # Libraries needed to run the project
```

---

## 🛠️ Technologies and Libraries Used

Python 3.x
Google Colab (development environment)
Pandas, NumPy (data manipulation and analysis)
Matplotlib, Seaborn, Plotly (visualization)
scikit-learn (modeling and evaluation)
XGBoost, LightGBM (boosting models)
imbalanced-learn (handling unbalanced data)
SHAP (model interpretability)
Requests (connecting to external APIs)
Jupyter Notebooks

---

## 📚Data

The dataset contains variables such as:

- Contract type (monthly, two-year)
- Internet connection type (fiber optic, DSL, no internet)
- Use of digital services (streaming, electronic invoicing)
- Payment methods (electronic check, card, automatic debit)
- Customer tenure (tenure)
- Technical support and online security variables
- Target variable: churn (abandonment)

---

## ⚖️ Models and Evaluation

The following models are implemented and compared:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Stacking Ensemble
- K-nearest neighbors 

Main metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

## 🔖 Model Comparison

By metrics:

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC AUC | Observations                                             |
| ---------------------- | -------- | --------- | ------ | -------- | ------- | -------------------------------------------------------- |
| **XGBoost**            | 0.783    | 0.575     | 0.701  | 0.631    | 0.845   | Good overall balance, excellent ROC AUC, and high recall |
| **LightGBM**           | 0.774    | 0.560     | 0.703  | 0.623    | 0.843   | Very similar to XGBoost, good recall and precision       |
| **LogisticRegression** | 0.744    | 0.511     | 0.786  | 0.620    | 0.843   | Better recall (sensitivity), lower precision             |
| **RandomForest**       | 0.776    | 0.560     | 0.722  | 0.631    | 0.841   | High recall, good overall balance                        |
| **Stacking Ensemble**  | 0.779    | 0.578     | 0.623  | 0.600    | 0.834   | Better precision, slightly lower recall and F1           |
| **KNN**                | 0.714    | 0.475     | 0.725  | 0.574    | 0.794   | Lower accuracy and precision, but acceptable recall      |


According to Confusion Matrix:

| Model | TN | FP | FN | TP | Observations |
| ----------------------- | --- | --- | --- | --- | ------------------------------------------------------------------------------ |
| **KNN** | 735 | 300 | 103 | 271 | Good Recall (high TP) <br> Many False Positives (high FP) |
| **LightGBM** | 828 | 207 | 111 | 263 | Good Balance <br> Somewhat more FN than KNN, but less FP |
| **XGBoost** | 841 | 194 | 112 | 262 | Better TN (low FP) <br> FN similar to LightGBM |
| **RandomForest** | 823 | 212 | 104 | 270 | Balanced <br> FN slightly better than XGBoost |
| **LogisticRegression** | 754 | 281 | 80 | 294 | Better FN (fewer false negatives) <br> More FP (risk of higher costs) |
| **Ensemble (Stacking)** | 865 | 170 | 141 | 233 | Better TN (less FP) <br> More FN (risk of losing churning customers) |

🎯 Selected model: **XGBoost** as it delivers a robust, well-balanced model with good predictive power, ideal for minimizing both customer loss (due to high recall) and avoiding false alarms (due to reasonable accuracy). It's a model that adapts well to most business cases where the trade-off between capturing churn and not overreacting must be managed.

---

## 📊 Visualization Example

#### Comparative ROC curves of all models
<img width="691" height="547" alt="Comparative ROC curves of all models" src="https://github.com/user-attachments/assets/541c41fa-bb36-46cb-be60-025a9a232b88" />

#### Model comparison and classification metrics
<img width="846" height="547" alt="Model Comparison - Classification Metrics" src="https://github.com/user-attachments/assets/ed444caf-5b59-4f5a-b207-1a1658513570" />

#### Model Performance Comparison
<img width="863" height="690" alt="Performance comparison between models" src="https://github.com/user-attachments/assets/19e02b4f-d082-47f6-b0b9-97bd8afe3c41" />

#### Comparative Precision-Recall Curves for All Models
<img width="691" height="547" alt="Comparative Precision-Recall Curves of All Models" src="https://github.com/user-attachments/assets/cce4ffa5-f0e0-4dba-af15-330805b5dad7" />

---

## 🛠️ Instructions for Running the Notebook

1. Clone or download this repository:

```bash
git clone https://github.com/user/project-churn.git
```
2. Install the necessary dependencies (recommended: use a virtual environment):

```bash
pip install pandas matplotlib seaborn numpy plotly math matplotlib requests
```
3. Open the notebook in Jupyter, VSCode, or Google Colab:

4. Run the cells sequentially to replicate the full analysis.

---

## 📂 Project Access

- [Notebook in English](./TelecomX_LATAM2_en.ipynb)

---

##Author
**CYNTHIA YAEL MICHINSKI**

[LinkedIn](https://www.linkedin.com/in/michinski-cynthia-yael-a5b04417/) | [GitHub](https://github.com/accountCynthia14)

![Status](https://img.shields.io/badge/status-finished-brightgreen) 📅 Enero 2026

![Python](https://img.shields.io/badge/python-3.10-blue)

![NumPy](https://img.shields.io/badge/numpy-1.26.0-blue)

![Pandas](https://img.shields.io/badge/pandas-2.1.0-blue)

## Acknowledgments

<img width="180" height="180" alt="Screenshot 2025-08-13 034705" src="https://github.com/user-attachments/assets/bdfa03bc-d44a-4848-b622-6bac4e2dbc95" />

<img width="180" height="180" alt="images" src="https://github.com/user-attachments/assets/8ca15294-1738-45a7-af65-7a390e468937" />

<img width="180" height="180" alt="Oracle-Next-Education--e16783040" src="https://github.com/user-attachments/assets/8912c5a0-58d7-45af-ba13-d2a9f42cde5a" />
