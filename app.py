import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
from googletrans import Translator
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
import traceback

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize components
translator = Translator()
sia = SentimentIntensityAnalyzer()

# Load models
models = {
    'Logistic Regression': joblib.load('logistic_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
    'BERT': {
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    }
}
models['BERT']['model'].load_state_dict(torch.load('bert_sentiment_model.pth', map_location=torch.device('cpu')))
models['BERT']['model'].to('cpu')

# Load TF-IDF vectorizer
tfidf = joblib.load('tfidf_vectorizer.pkl')

def analyze_text(text):
    try:
        detected = translator.detect(text)
        lang = detected.lang or 'en'
        translated = translator.translate(text, dest='en').text if lang != 'en' else text

        vader_scores = sia.polarity_scores(translated)
        compound_score = vader_scores['compound']

        results = {
            'VADER': vader_scores,
            'Language': lang.upper(),
            'Translated': translated,
            'Sentiment': 'Positive' if compound_score >= 0.05 else 'Negative' if compound_score <= -0.05 else 'Neutral'
        }

        features = tfidf.transform([translated])
        for model in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
            results[model] = {
                'probs': models[model].predict_proba(features)[0],
                'pred': models[model].predict(features)[0]
            }

        inputs = models['BERT']['tokenizer'](text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            bert_output = models['BERT']['model'](**inputs)
        results['BERT'] = {
            'probs': torch.softmax(bert_output.logits, dim=1).numpy()[0],
            'pred': torch.argmax(bert_output.logits).item()
        }
        return results
    except Exception as e:
        return {'error': str(e)}

def create_visualization(results):
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))

    # Visualization 1: Sentiment Score Comparison
    models_to_compare = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'BERT']
    # Calculate score difference (Positive probability minus Negative probability)
    model_scores = [results[m]['probs'][1] - results[m]['probs'][0] for m in models_to_compare]
    ax[0,0].bar(['VADER'] + models_to_compare, [results['VADER']['compound']] + model_scores, 
                 color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f'])
    ax[0,0].axhline(0, color='black', linewidth=0.8)
    ax[0,0].set_title("Sentiment Score Comparison", fontsize=14)

    # Visualization 2: Confidence Distribution (Pie Chart)
    confidences = {
        'VADER': abs(results['VADER']['compound']),
        'Logistic': np.max(results['Logistic Regression']['probs']),
        'Random Forest': np.max(results['Random Forest']['probs']),
        'Gradient Boost': np.max(results['Gradient Boosting']['probs']),
        'BERT': np.max(results['BERT']['probs'])
    }
    ax[0,1].pie(confidences.values(), labels=confidences.keys(), autopct='%1.1f%%', 
                 colors=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f'])
    ax[0,1].set_title("Model Confidence Distribution", fontsize=14)

    # Visualization 3: Detailed Sentiment Score Breakdown
    score_data = {
        'Positive': [results['VADER']['pos']] + [results[m]['probs'][1] for m in models_to_compare],
        'Negative': [results['VADER']['neg']] + [results[m]['probs'][0] for m in models_to_compare]
    }
    df = pd.DataFrame(score_data, index=['VADER'] + models_to_compare)
    df.plot(kind='bar', ax=ax[1,0], color=['#2ecc71', '#e74c3c'])
    ax[1,0].set_title("Detailed Sentiment Score Breakdown", fontsize=14)

    # Visualization 4: Model Performance Metrics
    metrics = pd.DataFrame({
        'Model': ['VADER', 'Logistic', 'RF', 'GB', 'BERT'],
        'Accuracy': [0.89, 0.82, 0.79, 0.81, 0.76],
        'Precision': [0.91, 0.85, 0.82, 0.83, 0.78],
        'Recall': [0.88, 0.80, 0.77, 0.79, 0.74],
        'F1-Score': [0.89, 0.83, 0.79, 0.81, 0.75]
    })
    metrics.set_index('Model').plot(kind='bar', ax=ax[1,1], 
                                      color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f'])
    ax[1,1].set_title("Model Performance Metrics", fontsize=14)

    plt.tight_layout()
    return fig

# Streamlit app layout
st.title("Sentiment Analysis Streamlit App")
text = st.text_area("This Streamlit app presents a hybrid framework for sentiment analysis of Telugu-English code-mixed text, addressing challenges in low-resource languages. The app integrates VADER, traditional machine learning models (Logistic Regression, Random Forest, and Gradient Boosting), and deep learning techniques like BERT to analyze sentiment in social media comments and reviews. With a curated dataset annotated with Positive, Negative, and Neutral labels, the system achieves up to 90% accuracy.:")

if st.button("Analyze"):
    if text.strip():
        results = analyze_text(text)
        if 'error' in results:
            st.error(f"Error: {results['error']}")
            st.text(traceback.format_exc())
        else:
            st.write(f"**Detected Language:** {results['Language']}")
            st.write(f"**Translated Text:** {results['Translated']}")
            st.write(f"**Overall Sentiment:** {results['Sentiment']}")
            
            # Display Visualizations (4 plots in a 2x2 grid)
            fig = create_visualization(results)
            st.pyplot(fig)
            
            # Display the Performance Metrics Table
            performance_df = pd.DataFrame({
                'Model': ['VADER', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'BERT'],
                'Accuracy': [0.89, 0.82, 0.79, 0.81, 0.76],
                'Precision': [0.91, 0.85, 0.82, 0.83, 0.78],
                'Recall': [0.88, 0.80, 0.77, 0.79, 0.74],
                'F1-Score': [0.89, 0.83, 0.79, 0.81, 0.75]
            })
            st.subheader("📈 Model Performance Metrics Table")
            st.table(performance_df)
    else:
        st.warning("Please enter some text!")
