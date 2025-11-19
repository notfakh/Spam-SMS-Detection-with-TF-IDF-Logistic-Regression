# Spam-SMS-Detection-with-TF-IDF-Logistic-Regression
Builds a spam detection system using a Logistic Regression classifier on preprocessed SMS messages, vectorized with TF-IDF, and evaluates model performance with accuracy and a classification report.

## 📋 Project Overview

This project implements a binary text classifier to identify spam messages from SMS data. It uses natural language processing techniques including text preprocessing, stopword removal, TF-IDF vectorization, and Logistic Regression for classification.

## 🎯 Problem Statement

**Can we automatically detect spam messages based on their text content?**

This classifier analyzes SMS messages and predicts whether they are:
- **Ham (0)**: Legitimate messages
- **Spam (1)**: Unwanted promotional or fraudulent messages

## 🔑 Key Features

- ✅ Text preprocessing with NLTK
- ✅ HTML tag removal
- ✅ Special character filtering
- ✅ Stopword removal (English)
- ✅ TF-IDF vectorization (5000 features)
- ✅ Logistic Regression classification
- ✅ Comprehensive performance metrics
- ✅ Binary classification (Ham vs Spam)

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notfakh/sms-spam-classifier.git
cd sms-spam-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download `spam.csv` from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   - Place it in the project root directory

### Usage

Run the script:
```bash
python spam_classifier.py
```

**First Run:**
- Automatically downloads NLTK stopwords
- Preprocesses all text messages
- Trains the model
- Displays classification report and accuracy

## 📊 Text Preprocessing Pipeline

### Step-by-Step Process:

1. **HTML Tag Removal**
   ```python
   "<p>Hello</p>" → "Hello"
   ```

2. **Special Character Filtering**
   ```python
   "Call now! #1 offer @ $99" → "Call now offer"
   ```

3. **Lowercase Conversion**
   ```python
   "FREE MONEY" → "free money"
   ```

4. **Stopword Removal**
   ```python
   "this is a spam message" → "spam message"
   ```

5. **Tokenization & Cleaning**
   - Splits text into words
   - Removes common English words (the, is, at, etc.)
   - Joins cleaned words back together

## 📈 Model Architecture

### TF-IDF Vectorization
```python
TfidfVectorizer(
    max_features=5000,    # Top 5000 most important words
    stop_words='english'  # Remove common English words
)
```

**What is TF-IDF?**
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How unique the word is across all documents
- **Result**: Words that are frequent but unique get higher scores

### Logistic Regression Classifier
```python
LogisticRegression(max_iter=1000)
```
- Binary classification algorithm
- Outputs probability of being spam
- Threshold: 0.5 (>0.5 = Spam, <0.5 = Ham)

## 📊 Expected Results

### Sample Performance Metrics:

| Metric | Ham | Spam |
|--------|-----|------|
| **Precision** | ~0.99 | ~0.96 |
| **Recall** | ~0.99 | ~0.90 |
| **F1-Score** | ~0.99 | ~0.93 |
| **Overall Accuracy** | ~97-98% |

### Classification Report Output:
```
              precision    recall  f1-score   support

         Ham       0.99      0.99      0.99       965
        Spam       0.96      0.90      0.93       150

    accuracy                           0.98      1115
   macro avg       0.97      0.95      0.96      1115
weighted avg       0.98      0.98      0.98      1115
```

## 🔍 Dataset Information

**SMS Spam Collection Dataset:**
- **Total Messages**: 5,572 SMS messages
- **Ham Messages**: ~87% (4,827 messages)
- **Spam Messages**: ~13% (747 messages)
- **Format**: CSV with two columns (v1: label, v2: text)
- **Source**: UCI Machine Learning Repository

**Column Mapping:**
- `v1` → `sentiment` (spam/ham label)
- `v2` → `review` (SMS text content)

## 🛠️ Customization

### Adjust TF-IDF Features

Change the number of features:
```python
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
```

### Try Different Classifiers

Replace Logistic Regression:
```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

# Or Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
```

### Modify Preprocessing

Add stemming or lemmatization:
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    # ... existing preprocessing ...
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
```

### Change Train/Test Split

Adjust the split ratio:
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], 
    test_size=0.3,  # 30% for testing
    random_state=42
)
```

## 💡 Understanding the Metrics

### Precision
- **Ham Precision**: Of all messages classified as ham, how many were actually ham?
- **Spam Precision**: Of all messages classified as spam, how many were actually spam?

### Recall
- **Ham Recall**: Of all actual ham messages, how many did we correctly identify?
- **Spam Recall**: Of all actual spam messages, how many did we catch?

### F1-Score
- Harmonic mean of precision and recall
- Balances both metrics
- Higher is better (max = 1.0)

### Accuracy
- Overall percentage of correct predictions
- (True Positives + True Negatives) / Total Predictions

## 🔬 Extending the Project

Ideas for enhancement:

1. **Cross-Validation**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(clf, X_train_vec, y_train, cv=5)
   ```

2. **Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'C': [0.1, 1, 10], 'max_iter': [500, 1000]}
   grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
   ```

3. **Confusion Matrix Visualization**
   ```python
   from sklearn.metrics import confusion_matrix
   import seaborn as sns
   cm = confusion_matrix(y_test, y_pred)
   sns.heatmap(cm, annot=True, fmt='d')
   ```

4. **Word Cloud Visualization**
   ```python
   from wordcloud import WordCloud
   spam_words = ' '.join(df[df['sentiment']==1]['review'])
   wordcloud = WordCloud().generate(spam_words)
   ```

5. **Save and Load Model**
   ```python
   import pickle
   pickle.dump(clf, open('spam_classifier.pkl', 'wb'))
   model = pickle.load(open('spam_classifier.pkl', 'rb'))
   ```

6. **Real-time Prediction Function**
   ```python
   def predict_spam(message):
       processed = preprocess_text(message)
       vectorized = vectorizer.transform([processed])
       prediction = clf.predict(vectorized)[0]
       return "Spam" if prediction == 1 else "Ham"
   ```

## 📊 Feature Importance

View most important words for classification:
```python
feature_names = vectorizer.get_feature_names_out()
coefficients = clf.coef_[0]
top_spam = sorted(zip(coefficients, feature_names), reverse=True)[:10]
print("Top spam indicators:", top_spam)
```

## 🤝 Contributing

Contributions welcome! Enhancement ideas:

- Add deep learning models (LSTM, BERT)
- Implement n-gram analysis
- Create web interface with Flask/Streamlit
- Add multilingual support
- Include confidence scores
- Implement active learning
- Add model explainability (LIME, SHAP)

## 👤 Author

**Fakhrul Sufian**
- GitHub: [@notfakh](https://github.com/notfakh)
- LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- NLTK library for natural language processing tools
- Scikit-learn for machine learning implementations
- Kaggle community for dataset hosting

## 📚 References

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [NLTK Documentation](https://www.nltk.org/)

## 🐛 Troubleshooting

**Issue: NLTK stopwords not found**
```python
import nltk
nltk.download('stopwords')
```

**Issue: Encoding error**
- The script uses `encoding='ISO-8859-1'` for compatibility
- Try `encoding='utf-8'` if issues persist

**Issue: Low accuracy**
- Increase `max_features` in TfidfVectorizer
- Try different classifiers (SVM, Random Forest)
- Add n-grams: `ngram_range=(1,2)`

**Issue: Imbalanced dataset**
- Use `class_weight='balanced'` in LogisticRegression
- Apply SMOTE for oversampling minority class

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue in this repository
- Email: fkhrlnasry@gmail.com
- Connect on LinkedIn

---

⭐ If this project helped you understand text classification and spam detection, please give it a star!

## 🎓 Learning Outcomes

After working through this project, you'll understand:
- Text preprocessing techniques with NLTK
- TF-IDF vectorization for text representation
- Binary classification with Logistic Regression
- Evaluation metrics for classification tasks
- Handling imbalanced datasets
- Natural language processing pipelines

**Perfect for:** NLP beginners, data science students, and anyone interested in text classification!
