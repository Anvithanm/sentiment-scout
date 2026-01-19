# The Sentiment Scout  
**Aspect-Based Sentiment Analysis of Customer Experience**

## Overview  
**Sentiment Scout** is an end-to-end **Aspect-Based Sentiment Analysis (ABSA)** system designed to extract actionable insights from unstructured customer reviews. The project demonstrates the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, and deployment as an interactive **Streamlit web application**.

The system identifies **sentiment polarity (positive, neutral, negative)** and the **specific aspects** (e.g., food, service, staff, ambience) discussed in customer feedback, enabling fine-grained analysis beyond document-level sentiment.

This project was built primarily in **Jupyter Notebook** for experimentation and analysis, and later deployed as a production-style application, reflecting applied ML engineering practices.

---

## Key Highlights (ML Engineer Focus)
- End-to-end NLP pipeline implemented in **Jupyter Notebook**
- Large-scale real-world dataset (~33K customer reviews)
- Aspect extraction using **spaCy POS tagging**
- TF-IDF feature representation for classical ML models
- Comparative modeling with **Multinomial Naive Bayes** and **SVM**
- Quantitative evaluation using Accuracy, Precision, Recall, and F1-score
- Deployment using **Streamlit** for real-time inference

---

## Dataset  
- **Source**: Google Reviews (McDonald’s stores across the United States)  
- **Size**: 33,396 customer reviews  
- **Core Features Used**:
  - `review` – textual customer feedback  
  - `rating` – star rating (1–5)

---

## Methodology  

### 1. Data Preprocessing  
Text preprocessing was performed using **NLTK** and **spaCy**, including:
- Duplicate and missing value removal  
- Case normalization  
- Noise removal using regular expressions  
- Stop-word removal  
- Tokenization  
- Lemmatization  

---

### 2. Feature Engineering  

**Sentiment Labeling**
- Review-based sentiment derived using **VADER (NLTK)**  
- Rating-based sentiment derived using thresholds:
  - Positive: 4–5 stars  
  - Neutral: 3 stars  
  - Negative: 1–2 stars  

**Aspect Extraction**
- Extracted using **spaCy POS tagging**
- Nouns and proper nouns treated as candidate aspects
- Generated aspect-level features per review

---

### 3. Feature Representation  
- Text converted into numerical vectors using **TF-IDF**
- Captures term importance while reducing noise from common words

---

### 4. Model Training  
The following supervised models were trained and compared:
- **Multinomial Naive Bayes**
  - Case 1: Trained on review-derived sentiment  
  - Case 2: Trained on rating-derived sentiment  
- **Support Vector Machine (SVM)**

---

### 5. Model Evaluation  
Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

**Selected Model for Deployment**
- **Multinomial Naive Bayes (Rating-based sentiment)**
- **Accuracy**: **79.34%**
- Chosen for balanced performance and interpretability

---

## Web Application  
The trained model is deployed as an interactive **Streamlit** application.

**Application Features**
- User input for custom reviews  
- Real-time sentiment prediction  
- Extracted aspects displayed alongside sentiment  

**Tech Stack**
- Python  
- Streamlit  
- scikit-learn  
- NLTK  
- spaCy  

---

## Repository Structure  
```
├── Application-sentiment-scout.py   # Streamlit web application
├── Image_app.png                    # App UI snapshot
├── sentiment_analysis.pkl           # Serialized trained model
├── utils.py                         # Helper functions (preprocessing, inference)
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── __init__.py
│
├── Model/
│   └── Sentiment_Analysis_Model.ipynb   # End-to-end ML pipeline (EDA → modeling)
│
├── dataset/
│   ├── McDonalds_Reviews.csv            # Raw dataset
│   └── project_processed_data.csv      # Cleaned & processed dataset
│
├── docs/
│   ├── Building_block.jpg               # Model architecture overview
│   ├── Image_app.png                    # Application screenshot
│   ├── Presentation2.pptx               # Project presentation
│   └── Project_Report.pdf
```

---

## Installation & Usage  

```bash
git clone <repository-url>
cd sentiment-scout
pip install -r requirements.txt
streamlit run app.py
```

---

## Results & Impact  
- Demonstrates applied **machine learning and NLP engineering**
- Translates research workflows into deployable applications
- Highlights modeling trade-offs and evaluation rigor
- Applicable to customer experience analytics and feedback mining

---

## Future Improvements  
- Transformer-based models (BERT, RoBERTa)
- Aspect–sentiment pairing at sentence level
- Improved neutral-class recall
- Model monitoring and retraining pipeline

---

## Author  
**Anvitha Hiriadka**  
MS in Information Systems  
Northeastern University  
April 2024  
