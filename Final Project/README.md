# 📝 Sentiment Analysis Project – TYM Fellowship 2025 (AI/ML Track)

## 📖 Project Description
This project implements a **Sentiment Analysis system** using both **Classic Machine Learning models** and a **Modern Large Language Model (LLM)** (Google Gemini).  
It is the final project for the **TYM Fellowship 2025 – AI/ML Track**.

The workflow is divided into two main approaches:

1. **Classic ML Classifier** (Traditional NLP + ML models)  
2. **Modern LLM Analyzer** (Gemini API for sentiment classification)

By comparing these two methods, we gain insights into the trade-offs between conventional supervised learning techniques and advanced LLM-based analysis.

---

## ⚙️ Implementation Details

### 🔹 Dataset
We use the **Sentiment Labelled Sentences Dataset** (UCI Machine Learning Repository).  
It contains **3 sources** of 1,000 sentences each (labeled `1` for positive, `0` for negative):
- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

The dataset is provided as a `.zip` file and extracted in the notebook.

---

### 🔹 Classic ML Classifier (Section 4.1.1 in project PDF)
Steps followed:
1. **Data Loading & Preprocessing**
   - Load all three datasets separately (Amazon, IMDb, Yelp).
   - Clean text (basic normalization).

2. **Feature Engineering**
   - Apply **TF-IDF Vectorization** to convert text into numerical features.

3. **Model Training**
   - Train multiple ML classifiers:
     - Logistic Regression  
     - Naïve Bayes  
     - Support Vector Machine (SVM)  
     - Random Forest  

4. **Evaluation**
   - Evaluate accuracy, precision, recall, and F1-score.
   - Compare performance across models.

---

### 🔹 Modern LLM Analyzer – Gemini (Section 5.2 in project PDF)
Steps followed:
1. **API Configuration**
   - Gemini API configured using environment variable:  
     `GOOGLE_API_KEY`

2. **Sentiment Analysis**
   - Sentences are sent as prompts to Gemini (`models/gemini-1.5-flash` / `gemini-1.5-pro`).  
   - Model outputs `Positive (1)` or `Negative (0)` labels.

3. **Evaluation**
   - Gemini’s predictions are compared against ground-truth labels.
   - Accuracy is calculated on a subset of the dataset.

---

### 🔹 Final Comparison
- Classic ML best-performing model (e.g., **SVM or Logistic Regression**) is compared against **Gemini**.  
- Results are summarized in a table:

| Approach          | Accuracy | Notes |
|-------------------|----------|-------|
| Logistic Regression (Classic ML) | ~XX% | Requires preprocessing & feature engineering |
| Gemini LLM (Modern) | ~YY% | Few-shot, more robust but API-dependent |

---

## 📦 File Structure
project/
│── final_project_sentiment.ipynb  # Main Jupyter Notebook
│── sentiment_labelled_sentences.zip  # Dataset (uploaded in Colab)
│── README.md  # This file

---

## ✅ Acceptance Criteria
- ✔️ Dataset loaded and preprocessed  
- ✔️ Multiple ML models trained and evaluated  
- ✔️ Gemini API integrated and tested  
- ✔️ Final comparison performed (Classic ML vs LLM)  
- ✔️ No runtime errors in notebook  

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**.
2. Upload the dataset ZIP file (`sentiment labelled sentences.zip`) when prompted.
3. Run all cells in sequence:
   - Classic ML section
   - Modern LLM Analyzer (Gemini API key required)
4. View the final comparison results.

---

## 🔑 Requirements
- Python 3.10+  
- Google Colab environment (recommended)  
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `google-generativeai`

Install them in Colab with:
```bash
!pip install pandas scikit-learn google-generativeai
