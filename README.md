# ReneWind: Wind Turbine Generator Failure Prediction

A neural network classification project that predicts wind turbine generator failures from sensor data, so that maintenance can be planned before a breakdown happens.

## Problem

ReneWind collects readings from sensors fitted across its wind turbines. A missed failure is expensive, because the whole generator has to be replaced. A false alarm only costs an inspection.

| Outcome | What happens | Cost |
|---|---|---|
| True Positive | Failure predicted correctly, so the part is repaired | Low |
| False Negative | Failure missed, so the generator is replaced | **High** |
| False Positive | No failure, but the turbine is flagged and inspected | Medium |

Because missed failures cost the most, **Recall** is the primary metric. **F1-score** is tracked as a secondary metric.

## Dataset

- `Train.csv`: 20,000 rows, 40 sensor features (V1 to V40) plus a binary `Target` (1 = failure)
- `Test.csv`: 5,000 rows with the same columns
- Highly imbalanced: only about 5.55% of observations are failures
- Missing values: a handful in V1 and V2 only (36 in train, 11 in test)

## Approach

1. **EDA:** target distribution, feature histograms, box plots for outliers, class-wise feature means, correlation heatmap, and correlation with the target.
2. **Preprocessing:** median imputation, a stratified 80/20 train/validation split, and `StandardScaler`. The medians and the scaler are fitted on training data only, which avoids data leakage.
3. **Class imbalance:** class weights computed with scikit-learn and a lowered decision threshold of 0.3 instead of the default 0.5.
4. **Modelling:** seven Keras `Sequential` networks, each changing one idea from the previous one (optimizer, depth, dropout, class weights, batch normalisation). All use early stopping on validation loss.
5. **Selection and testing:** the best model is chosen by validation recall and then evaluated once on the held-out test set.

## Models compared

Validation set, threshold = 0.3:

| # | Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | Baseline, SGD (64-32) | 0.910 | 0.944 | 0.927 | 0.954 |
| 2 | Deeper, SGD + momentum | 0.919 | 0.944 | 0.932 | 0.966 |
| 3 | Adam | 0.914 | 0.967 | 0.940 | 0.962 |
| 4 | Adam + Dropout | 0.914 | 0.967 | 0.940 | 0.970 |
| 5 | **Adam + Class Weights** | **0.923** | 0.262 | 0.409 | 0.944 |
| 6 | SGD + Dropout + Class Weights | 0.923 | 0.271 | 0.419 | 0.944 |
| 7 | Adam + Deep + Dropout + BatchNorm + Class Weights | 0.919 | 0.389 | 0.546 | 0.953 |

## Final results (test set)

Model 5 (Adam + Class Weights), threshold = 0.3:

| Metric | Score |
|---|---|
| Recall | 0.879 |
| Precision | 0.270 |
| F1-score | 0.414 |
| ROC-AUC | 0.924 |

The model catches about 88% of real failures (248 of 282). The trade-off is a high number of false alarms (precision is 27%), which is acceptable here because an unnecessary inspection costs far less than a missed failure.

## Key takeaways

- Class imbalance and the decision threshold have a large effect on how many failures are caught and how many false alarms are raised.
- Class weights lift recall slightly but cut precision sharply, so the model that is best on recall is not the best on F1.
- Recall dropped from 0.92 on validation to 0.88 on test, so some of the validation performance did not carry over to unseen data.

## Tech stack

Python, TensorFlow / Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook

## How to run

1. Clone the repo and place `Train.csv` and `Test.csv` in the same folder as the notebook.
2. Open `ReneWind_NeuralNetwork_Project.ipynb` in Jupyter or Google Colab.
3. Run all cells from top to bottom. The first cell installs the required packages, and the plots are saved as `.png` files in the same folder.

## Author

**Vivaswan Tiwari**
[LinkedIn](https://www.linkedin.com/in/vivaswan-tiwari39853b140) · [GitHub](https://github.com/viva2601)
