Sentiment Analysis on Twitter US Airline Reviews

This project focuses on sentiment analysis using the Twitter US Airline Sentiment Dataset.
The goal is to classify tweets into three categories — Positive, Neutral, or Negative — based on user opinions about various airlines.

The project reawakens NLP, preprocessing, and model evaluation skills, while adding interpretability and visualization.

Objectives:
- Clean and preprocess real-world tweet text.
- Extract meaningful numerical representations using TF-IDF.
- Train and evaluate Logistic Regression and Naive Bayes classifiers.
- Visualize model performance with interactive Plotly charts.
- Interpret model decisions through top positive and negative words.

Dataset:
- Source: Twitter US Airline Sentiment Dataset (Kaggle)

Description:
- The dataset contains tweets about major US airlines, each labeled with a sentiment category (positive, neutral, or negative), along with the airline name, tweet text, and metadata.

Results Summary:
- Model	Accuracy	Precision	Recall	F1-Score
- Logistic Regression	~0.82	High precision across all classes	Balanced recall	Strong overall
- Naive Bayes	~0.78	Slightly higher recall for negatives	Good interpretability	Fast and efficient

Top Positive Words (LR): great, love, awesome, best, thank
Top Negative Words (LR): delay, cancelled, terrible, bad, wait

Visual Insights:
- Interactive confusion matrices reveal misclassifications (mostly between neutral and positive).
- Word clouds and coefficient plots show clear sentiment separation cues.
- Feature explainability demonstrates how linguistic tone impacts predictions.

Key Learnings:
- Effective text cleaning drastically improves sentiment classification.
- Logistic Regression captures sentiment direction; Naive Bayes focuses on frequency patterns.
- Visualization + explainability transforms raw metrics into actionable insights.
- Plotly elevates interpretability for both technical and non-technical audiences.

Future Goal:
- Try fine-tuning a pretrained BERT model using the same dataset.
- Compare deep learning-based embeddings with TF-IDF for richer sentiment context.

Tools & Libraries:
- Python, Pandas, NumPy
- scikit-learn
- nltk (for stopwords & lemmatization)
- Plotly (interactive visuals)
- Matplotlib / Seaborn

Conclusion:

This mini-project demonstrates the complete NLP workflow — from preprocessing to explainability.
Both models perform well, but Logistic Regression offers a slight edge in accuracy and interpretability, while Naive Bayes remains faster and robust for real-time sentiment pipelines.

Authour:
- Chukwuemeka Eugene Obiyo
- Data Scientist | Machine Learning Engineer
- praise609@gmail.com
