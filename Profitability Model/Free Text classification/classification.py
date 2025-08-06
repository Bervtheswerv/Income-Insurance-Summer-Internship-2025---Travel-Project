"""
Insurance Accident Description Classification System
Complete implementation for processing and classifying insurance accident descriptions
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core NLP packages
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# ============================================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================================

def preprocess_text(texts):
    """
    Basic text preprocessing for accident descriptions
    """
    import re
    
    processed = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        processed.append(text)
    
    return processed

# ============================================================================
# STEP 3: UNSUPERVISED PATTERN DISCOVERY
# ============================================================================

def discover_topics(df, n_topics='auto'):
    """
    Use BERTopic to discover patterns in accident descriptions
    """
    print("\n" + "="*60)
    print("STEP 1: UNSUPERVISED PATTERN DISCOVERY")
    print("="*60)
    
    # Preprocess texts
    texts = df['accident_description'].tolist()
    processed_texts = preprocess_text(texts)
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure BERTopic
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5
    )
    
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        n_gram_range=(1, 2),
        min_topic_size=30,
        nr_topics=n_topics,
        verbose=True
    )
    
    # Fit the model
    print("\nFitting BERTopic model...")
    topics, probs = topic_model.fit_transform(processed_texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info)-1} topics (excluding outliers)")
    
    # Display top topics
    print("\nTop 10 Topics:")
    for idx, row in topic_info.head(11).iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            print(f"\nTopic {row['Topic']}: {row['Count']} documents")
            # Get top words for this topic
            words = topic_model.get_topic(row['Topic'])
            top_words = [word[0] for word in words[:5]]
            print(f"Keywords: {', '.join(top_words)}")
    
    # Add topics to dataframe
    df['discovered_topic'] = topics
    df['topic_probability'] = probs
    
    return df, topic_model

# ============================================================================
# STEP 4: KEYWORD EXTRACTION
# ============================================================================

def extract_keywords(df, topic_model):
    """
    Extract key terms from accident descriptions
    """
    print("\n" + "="*60)
    print("STEP 2: KEYWORD EXTRACTION")
    print("="*60)
    
    # Initialize KeyBERT
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    
    # Extract keywords for each topic
    topic_keywords = {}
    
    for topic_id in df['discovered_topic'].unique():
        if topic_id != -1:  # Skip outliers
            # Get sample texts from this topic
            topic_texts = df[df['discovered_topic'] == topic_id]['accident_description'].head(100).tolist()
            combined_text = ' '.join(topic_texts)
            
            # Extract keywords
            keywords = kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=20
            )
            
            topic_keywords[topic_id] = keywords
            
            print(f"\nTopic {topic_id} key phrases:")
            for kw, score in keywords[:10]:
                print(f"  - {kw}: {score:.3f}")
    
    return topic_keywords

# ============================================================================
# STEP 5: CATEGORY MAPPING
# ============================================================================

def map_topics_to_categories(df, topic_model, topic_keywords):
    """
    Map discovered topics to meaningful categories
    """
    print("\n" + "="*60)
    print("STEP 3: CATEGORY MAPPING")
    print("="*60)
    
    # Analyze topic characteristics
    category_mapping = {}
    
    for topic_id in topic_keywords:
        keywords = [kw[0] for kw in topic_keywords[topic_id]]
        keywords_str = ' '.join(keywords).lower()
        
        # Rule-based mapping based on keywords
        if any(word in keywords_str for word in ['vehicle', 'car', 'collision', 'accident', 'driver', 'traffic']):
            category_mapping[topic_id] = 'Vehicle Accident'
        elif any(word in keywords_str for word in ['medical', 'hospital', 'illness', 'symptoms', 'treatment', 'doctor']):
            category_mapping[topic_id] = 'Medical Emergency'
        elif any(word in keywords_str for word in ['flight', 'luggage', 'travel', 'trip', 'airport', 'delayed']):
            category_mapping[topic_id] = 'Travel Incident'
        elif any(word in keywords_str for word in ['property', 'damage', 'fire', 'water', 'storm', 'burglary']):
            category_mapping[topic_id] = 'Property Damage'
        elif any(word in keywords_str for word in ['sport', 'injury', 'playing', 'exercise', 'activity']):
            category_mapping[topic_id] = 'Sports Injury'
        else:
            category_mapping[topic_id] = f'Other (Topic {topic_id})'
    
    # Apply mapping
    df['predicted_category'] = df['discovered_topic'].map(category_mapping)
    df.loc[df['discovered_topic'] == -1, 'predicted_category'] = 'Uncategorized'
    
    print("\nCategory Distribution:")
    print(df['predicted_category'].value_counts())
    
    return df, category_mapping

# ============================================================================
# STEP 6: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_clustering(df):
    """
    Evaluate clustering performance
    """
    print("\n" + "="*60)
    print("STEP 4: EVALUATION")
    print("="*60)
    
    # Compare with true categories
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Filter out outliers for fair comparison
    mask = df['discovered_topic'] != -1
    
    ari = adjusted_rand_score(
        df.loc[mask, 'true_category'],
        df.loc[mask, 'predicted_category']
    )
    
    nmi = normalized_mutual_info_score(
        df.loc[mask, 'true_category'],
        df.loc[mask, 'predicted_category']
    )
    
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    
    # Confusion matrix
    print("\nCategory Confusion Matrix:")
    true_categories = df['true_category'].unique()
    pred_categories = df['predicted_category'].unique()
    
    # Create a mapping for visualization
    category_conf = pd.crosstab(
        df['true_category'],
        df['predicted_category'],
        normalize='index'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(category_conf, annot=True, fmt='.2f', cmap='Blues')
    plt.title('True vs Predicted Category Confusion Matrix')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig('category_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'category_confusion_matrix.png'")
    
    return ari, nmi

# ============================================================================
# STEP 7: SUPERVISED CLASSIFICATION SETUP
# ============================================================================

def prepare_supervised_training(df, category_mapping):
    """
    Prepare data for supervised learning
    """
    print("\n" + "="*60)
    print("STEP 5: PREPARING SUPERVISED TRAINING DATA")
    print("="*60)
    
    # Use discovered categories as labels
    # In real scenario, these would be validated by domain experts
    
    # Remove outliers for supervised training
    train_df = df[df['discovered_topic'] != -1].copy()
    
    # Prepare features and labels
    X = train_df['accident_description'].tolist()
    y = train_df['predicted_category'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of categories: {len(set(y))}")
    
    # Save prepared data
    train_data = pd.DataFrame({
        'text': X_train,
        'category': y_train
    })
    test_data = pd.DataFrame({
        'text': X_test,
        'category': y_test
    })
    
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    print("\nTraining data saved to 'train_data.csv' and 'test_data.csv'")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 8: SAMPLE CLASSIFICATION PIPELINE
# ============================================================================

def create_classification_pipeline(X_train, X_test, y_train, y_test):
    """
    Create a simple classification pipeline for demonstration
    """
    print("\n" + "="*60)
    print("STEP 6: CLASSIFICATION PIPELINE")
    print("="*60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    print("Training classifier...")
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance (top words per category)
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    
    print("\nTop features per category:")
    for category in sorted(set(y_train)):
        # Get samples for this category
        category_mask = [y == category for y in y_train]
        category_texts = [x for x, mask in zip(X_train, category_mask) if mask]
        
        # Transform texts
        category_features = pipeline.named_steps['tfidf'].transform(category_texts)
        
        # Get top features
        feature_scores = category_features.mean(axis=0).A1
        top_indices = feature_scores.argsort()[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        
        print(f"\n{category}:")
        print(f"  {', '.join(top_features[:5])}")
    
    return pipeline

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*60)
    print("INSURANCE ACCIDENT DESCRIPTION CLASSIFICATION")
    print("="*60)
    
    # Generate dataset
    df = generate_accident_dataset(n_samples=10000)
    
    # Save raw dataset
    df.to_csv('accident_descriptions_raw.csv', index=False)
    print("\nRaw dataset saved to 'accident_descriptions_raw.csv'")
    
    # Discover topics
    df, topic_model = discover_topics(df)
    
    # Extract keywords
    topic_keywords = extract_keywords(df, topic_model)
    
    # Map to categories
    df, category_mapping = map_topics_to_categories(df, topic_model, topic_keywords)
    
    # Evaluate
    ari, nmi = evaluate_clustering(df)
    
    # Prepare supervised training
    X_train, X_test, y_train, y_test = prepare_supervised_training(df, category_mapping)
    
    # Create classification pipeline
    pipeline = create_classification_pipeline(X_train, X_test, y_train, y_test)
    
    # Save final results
    df.to_csv('accident_descriptions_processed.csv', index=False)
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print("Files generated:")
    print("- accident_descriptions_raw.csv: Original dataset")
    print("- accident_descriptions_processed.csv: Processed with topics and categories")
    print("- train_data.csv: Training data for supervised learning")
    print("- test_data.csv: Test data for supervised learning")
    print("- category_confusion_matrix.png: Evaluation visualization")
    
    # Save category mapping
    import json
    with open('category_mapping.json', 'w') as f:
        json.dump(category_mapping, f, indent=2)
    print("- category_mapping.json: Topic to category mapping")
    
    return df, topic_model, pipeline

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def classify_new_accidents(descriptions, topic_model, pipeline):
    """
    Classify new accident descriptions
    """
    print("\n" + "="*60)
    print("CLASSIFYING NEW DESCRIPTIONS")
    print("="*60)
    
    for i, desc in enumerate(descriptions):
        print(f"\nDescription {i+1}: {desc[:100]}...")
        
        # Get topic
        topic, prob = topic_model.transform([desc])
        print(f"Discovered topic: {topic[0]} (probability: {prob[0]:.3f})")
        
        # Get classification
        category = pipeline.predict([desc])[0]
        print(f"Predicted category: {category}")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    df, topic_model, pipeline = main()
    
    # Test with new examples
    new_descriptions = [
        "Hit by another car while waiting at red light. Severe whiplash and back pain. Other driver fled scene.",
        "Slipped on wet floor in hotel lobby. Fractured wrist. Required surgery and physiotherapy.",
        "Flight cancelled due to weather. Missed important business meeting. Lost $3000 in bookings."
    ]
    
    classify_new_accidents(new_descriptions, topic_model, pipeline)
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)