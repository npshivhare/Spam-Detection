# 📩 Spam Detection System using Machine Learning

A comparative spam detection system leveraging **Naive Bayes, Logistic Regression, and SVM**, combined with TF-IDF feature extraction for accurate SMS and email classification.

🚀 **Live App:** https://spam-detection-j38eizgf968q9jznkjclfa.streamlit.app

📂 **GitHub Repo:** https://github.com/npshivhare/Spam-Detection

---

## 📌 Overview

This project presents a **complete end-to-end spam detection pipeline** for classifying messages as spam or ham.

The system:
- Cleans and preprocesses raw text data  
- Converts text into numerical features using **TF-IDF (3000 features)**  
- Trains multiple machine learning models  
- Compares performance across classifiers  
- Saves trained models for future inference  

It provides a **robust, scalable, and deployable solution** for spam filtering.

---

## 🎯 Key Features

- 📩 SMS & Email spam classification  
- 🧠 ML models: Naive Bayes, Logistic Regression, SVM  
- 📊 TF-IDF based feature extraction  
- 📉 Confusion matrix & accuracy visualization  
- 💾 Model serialization using joblib  
- ⚡ Fast and efficient text preprocessing  
- 📈 Comparative performance analysis  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Scikit-learn (ML models & TF-IDF)  
  - Pandas, NumPy  
  - Matplotlib, Seaborn  
  - Joblib (Model saving)  
  - Regex, String (Preprocessing)  
- **Models:** Naive Bayes, Logistic Regression, SVM (Linear Kernel)

---

## ⚙️ System Architecture

1. Load dataset from ZIP file  
2. Preprocess text (lowercase, remove punctuation & digits)  
3. Convert text to TF-IDF vectors (3000 features)  
4. Split dataset (80% train / 20% test)  
5. Train ML models (NB, LR, SVM)  
6. Evaluate using accuracy & confusion matrix  
7. Save models and vectorizer (.pkl files)  

---

## 🧠 Model Selection

| Model               | Accuracy | Speed        | Robustness | Final Choice |
|--------------------|---------|-------------|------------|-------------|
| Naive Bayes        | 97.31%  | Very Fast   | High       | Strong Baseline |
| Logistic Regression| 95.52%  | Moderate    | Moderate   | Needs Tuning |
| SVM (Linear)       | 98.12%  | Slower      | Very High  | ⭐ Selected |

✅ **SVM chosen due to highest accuracy and best generalization performance**

---

## 📊 Methodology

### 🔹 Text Preprocessing
- Lowercasing  
- Removal of punctuation  
- Removal of digits  
- Stop-word removal  

### 🔹 Feature Extraction
- TF-IDF Vectorizer  
- Max Features: **3000**  

### 🔹 Model Training
- Naive Bayes (Probabilistic)  
- Logistic Regression (Linear Model)  
- SVM with Linear Kernel  

### 🔹 Evaluation
- Accuracy Score  
- Confusion Matrix  
- Performance Comparison Chart  

---

## 🖥️ Application Features

### 📌 Modules

- Data Loading → Extract & read dataset  
- Preprocessing → Clean text data  
- Feature Engineering → TF-IDF conversion  
- Model Training → Train 3 classifiers  
- Evaluation → Accuracy & confusion matrix  
- Model Saving → Export .pkl files  

---

## 📊 Results & Performance

- ✅ High accuracy across all models (>95%)  
- ⚡ Fast preprocessing and training pipeline  
- 🎯 SVM achieved best accuracy: **98.12%**  
- 📊 Naive Bayes performed well with minimal computation  
- 📉 Logistic Regression slightly lower due to default parameters  

---

## 🏆 Key Achievements

- Built **complete NLP-based spam detection pipeline**  
- Implemented **comparative study of 3 ML algorithms**  
- Achieved **98.12% accuracy using SVM**  
- Generated **automated confusion matrices & performance charts**  
- Enabled **model deployment via serialized .pkl files**  

---

## 👨‍💻 Team Members

- Krish Naik  
- Nrependre Shivhare  
- Prasham Godha  

---

## 🙏 Mentors

- Dr. K. K. Sharma  
- Dr. Lalit Purohit  
- Dr. Upendra Singh  
- Mr. Akshay Gupta  

---

## 🔮 Future Work

- Hyperparameter tuning (GridSearchCV)  
- Integration with real-time email/SMS systems  
- Deployment as web application (Flask/Streamlit)  
- Use of deep learning models (LSTM, BERT)  
- Multilingual spam detection support  

---

## 📚 References

- SMS Spam Collection Dataset (UCI Repository)  
- Scikit-learn Documentation  
- Research papers on Text Classification & Spam Detection  
- Naive Bayes, Logistic Regression, and SVM studies  

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
