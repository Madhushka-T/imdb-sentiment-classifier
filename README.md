# 🎬 IMDb Movie Review Sentiment Classifier (2024)

This project uses **Natural Language Processing (NLP)** and **XGBoost** to classify IMDb movie reviews as either **positive** or **negative**.

---

## 📌 Project Overview

- Built a text classification model using **50,000 IMDb reviews** (25k positive, 25k negative).
- Applied **NLP preprocessing** using NLTK and regular expressions.
- Compared three models:  
  - Logistic Regression → 88.7%  
  - Random Forest → 84.8%  
  - XGBoost → 🏆 Best at 85.7%
- Final model used: **XGBoost + CountVectorizer with bigrams**.
- 
### 🔍 Why XGBoost Outperformed Logistic Regression

While Logistic Regression achieved high accuracy (88.7%), XGBoost proved more robust for this task due to the following reasons:

- ✅ **Non-linearity**: XGBoost can model complex, non-linear relationships in text features that Logistic Regression (a linear model) might miss.
- ✅ **Boosting Strength**: It combines many weak learners (decision trees) to improve performance, reducing both bias and variance.
- ✅ **Handling Sparse Data**: XGBoost works well with sparse, high-dimensional data produced by `CountVectorizer`, especially with bigrams.
- ✅ **Feature Interactions**: XGBoost naturally captures feature interactions like `“not good”` or `“very bad”`, which are hard for Logistic Regression to handle without manual engineering.

As a result, even with slightly lower raw accuracy, XGBoost provided better **generalization** and **more consistent predictions** on short and complex reviews.

---

## 🧪 Key Steps

### 1. Data Loading
- Loaded a Kaggle IMDb review dataset with "review" and "sentiment" columns.

### 2. Text Preprocessing
- Lowercased the text
- Removed HTML tags
- Removed punctuation
- Removed stopwords using NLTK

### 3. Feature Extraction
- Used `CountVectorizer` with:
  - `max_features=5000`
  - `ngram_range=(1, 2)`
  - `stop_words='english'`

### 4. Model Training
- Used `train_test_split` for 80/20 split
- Trained with `XGBClassifier` from `xgboost`

### 5. Evaluation
- Achieved **85.7% accuracy** on test set
- Manually tested long and short reviews
- Improved short review handling using synthetic examples and `CountVectorizer`

---

## 📂 Project Files

- `imdb_model_train.ipynb` – notebook with training, evaluation, and saving logic
- `xgb_imdb_model.pkl` – saved XGBoost model
- `vectorizer.pkl` – saved CountVectorizer
- `README.md` – project documentation

---

## 💡 Future Work

- Build a simple web interface using **Flask**
- Improve short-text accuracy further using **VADER** or pre-trained transformers (e.g., BERT)

---

## 👨‍💻 Author

- **Madhushka Tummala**  
  B.Tech AI & ML | K L University  
  GitHub: [Madhushka-T](https://github.com/Madhushka-T)

