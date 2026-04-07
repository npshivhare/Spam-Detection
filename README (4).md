# 🛡️ Spam Detection AI — Streamlit App

A full-featured spam classifier dashboard comparing **Logistic Regression**, **Naïve Bayes**, and **SVM**.

## 📁 File Structure

```
your-folder/
├── app.py
├── requirements.txt
├── vectorizer.pkl                    ← rename from 1774471252513_vectorizer.pkl
├── lr_model.pkl                      ← rename from 1774471237053_Logistic_Regression_model.pkl
├── nb_model.pkl                      ← rename from 1774471237053_Naive_Bayes_model.pkl
└── svm_model.pkl                     ← rename from 1774471252513_SVM_model.pkl
```

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Rename your pkl files (or update the paths in app.py)
mv 1774471252513_vectorizer.pkl                    vectorizer.pkl
mv 1774471237053_Logistic_Regression_model.pkl     lr_model.pkl
mv 1774471237053_Naive_Bayes_model.pkl             nb_model.pkl
mv 1774471252513_SVM_model.pkl                     svm_model.pkl

# 3. Launch the app
streamlit run app.py
```

App opens at **http://localhost:8501**

## 🖥️ Features

| Page | Contents |
|------|----------|
| 🏠 Predict | Enter any text → get predictions from all 3 models with confidence bars |
| 📊 Algorithm Comparison | Accuracy / Precision / Recall / F1 / AUC table + radar chart |
| 🔍 Confusion Matrices | Heatmaps for TN/FP/FN/TP for each model |
| 🧠 Algorithm Guide | In-depth explanation of each algorithm + TF-IDF |
| 📈 Performance Charts | ROC curves, Precision-Recall curves, training time |
