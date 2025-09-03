# AI-ML-TYM-Fellowship
# TYM Fellowship 2025 – AI / ML Track

##  Fellowship Overview
The **TYM AI & ML Fellowship 2025** is an intensive and collaborative learning program designed to equip participants with practical skills in machine learning, neural networks, transformers, and API-driven LLM (Large Language Model) integrations.

During the fellowship, fellows engaged in a structured curriculum comprising:
- **Week-by-week sessions** on core ML topics (e.g., regression, classification, model evaluation).
- Hands-on learning with **CNNs**, **transformers**, and **prompt engineering**.
- Practical exposure to tools and APIs for real-time ML workflows.
- Guided mentorship and peer collaboration through projects and check-ins.

---

##  Final Project: Sentiment Analysis — Classic ML vs Modern LLM

### Project Objective
Build and evaluate two distinct sentiment classification systems using the UCI “Sentiment Labelled Sentences” dataset (Amazon, IMDb, Yelp). The goal is to compare the performance and trade-offs between:
1. **Classic Machine Learning Models**
2. **Modern LLM-Based Analysis (Google Gemini)**

---

###  Detailed Project Flow

| Stage                  | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Dataset Preparation** | Load three domain-specific text datasets and combine them for training.   |
| **Text Preprocessing**  | Clean, tokenize, and apply TF-IDF vectorization.                          |
| **Classic ML Models**   | Train Logistic Regression, Naïve Bayes, SVM, and Random Forest models.     |
| **LLM Sentiment Analysis** | Use Gemini API to infer sentiment for sample sentences.               |
| **Evaluation**          | Compare models using accuracy, precision, recall, F1-score, and confusion matrices. |
| **Final Comparison & Discussion** | Summarize findings and highlight performance, cost, and effort differences. |

---

##  How to Run This Project

### Recommended: Google Colab
1. Open `final_project_sentiment.ipynb` in Colab.
2. Upload or extract the dataset as instructed.
3. Add your `GOOGLE_API_KEY` via **Colab Secrets** — not in the notebook.
4. Run all cells sequentially:
    - Classic ML training & evaluation
    - Gemini LLM sentiment analysis
    - Final comparison summary

### Alternative: Local Environment
1. Clone the repository.
2. Install dependencies via:

    - pip install -r requirements.txt

3. Place dataset files inside `data/`.
4. Create a `.env` file with your API key (`GOOGLE_API_KEY=...`).
5. Run the notebook locally using Jupyter or VS Code.

---

##  Project Security Notes
Your notebook is safe for public sharing:
- The API key is never hardcoded into the code.
- It is securely loaded from `.env` files or Colab secrets.
- Outputs are cleared before uploading, ensuring no sensitive data or logs are exposed.

---

##  Final Thoughts
This project reflects the core skills and learning journey of the TYM AI/ML fellowship:
- Solid foundation in **classical ML pipelines**.
- Innovative integration of **modern LLM technologies**.
- Strong focus on reproducibility, security best practices, and real-world deployment considerations.

Congratulations on completing a truly professional-level capstone — this is exactly the caliber of work the fellowship aims to foster!

---

**Wasif Waheed**  
TYM Fellowship 2025 — AI / ML Track
