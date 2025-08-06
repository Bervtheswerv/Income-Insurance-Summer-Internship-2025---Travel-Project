# ============================================================================
# ACCIDENT DESCRIPTION NLP CLASSIFIER - JUPYTER NOTEBOOK STYLE
# Advanced classification using spaCy, scikit-learn, and semantic analysis
# ============================================================================

# BLOCK 1: INSTALL AND IMPORT PACKAGES
# ============================================================================
"""
First, install required packages:
pip install pandas numpy scikit-learn spacy nltk wordcloud matplotlib seaborn
python -m spacy download en_core_web_sm
pip install sentence-transformers textblob
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP packages
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

print("✅ All packages imported successfully!")

# ============================================================================
# BLOCK 2: LOAD AND EXPLORE DATA
# ============================================================================

# Load the data
input_file = '/Users/bervynwong/Desktop/INCOME Travel Insurance Portfolio Analysis Project/Profitability Model/Free Text classification/accident_description_raw.csv'
df = pd.read_csv(input_file)

print(f"📊 Dataset loaded: {len(df):,} records")
print(f"📋 Columns: {list(df.columns)}")
print(f"📄 Sample descriptions:")

# Display sample descriptions
for i in range(5):
    print(f"\n{i+1}. {df['AccidentDesc'].iloc[i]}")

# Check for missing values
print(f"\n🔍 Missing values in AccidentDesc: {df['AccidentDesc'].isnull().sum()}")
print(f"🔍 Empty descriptions: {(df['AccidentDesc'] == '').sum()}")

# Basic statistics
df['desc_length'] = df['AccidentDesc'].str.len()
print(f"\n📈 Description length stats:")
print(df['desc_length'].describe())

# ============================================================================
# BLOCK 3: ADVANCED TEXT PREPROCESSING WITH SPACY
# ============================================================================

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def advanced_text_preprocessing(text):
    """Advanced text preprocessing using spaCy"""
    if not isinstance(text, str):
        return ""
    
    # Process with spaCy
    doc = nlp(text.lower())
    
    # Extract meaningful tokens (remove stop words, punctuation, spaces)
    tokens = []
    medical_entities = []
    
    for token in doc:
        # Skip stop words, punctuation, and spaces
        if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 2:
            # Use lemmatized form
            tokens.append(token.lemma_)
        
        # Extract medical/injury related entities
        if token.pos_ in ['NOUN', 'ADJ'] and any(keyword in token.text.lower() for keyword in 
            ['pain', 'injury', 'sick', 'ill', 'hurt', 'ache', 'fever', 'stomach', 'head', 'knee', 'back']):
            medical_entities.append(token.text.lower())
    
    # Extract named entities (medical conditions, body parts)
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents 
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
    
    return {
        'cleaned_text': ' '.join(tokens),
        'medical_entities': medical_entities,
        'named_entities': entities,
        'token_count': len(tokens)
    }

# Apply preprocessing
print("🔄 Preprocessing text with spaCy...")
preprocessing_results = df['AccidentDesc'].apply(advanced_text_preprocessing)

# Extract results
df['cleaned_text'] = [result['cleaned_text'] for result in preprocessing_results]
df['medical_entities'] = [result['medical_entities'] for result in preprocessing_results]
df['token_count'] = [result['token_count'] for result in preprocessing_results]

print("✅ Text preprocessing completed!")
print(f"📊 Average tokens per description: {df['token_count'].mean():.1f}")

# Show preprocessing examples
print(f"\n📝 Preprocessing examples:")
for i in range(3):
    print(f"\nOriginal: {df['AccidentDesc'].iloc[i]}")
    print(f"Cleaned:  {df['cleaned_text'].iloc[i]}")
    print(f"Medical entities: {df['medical_entities'].iloc[i]}")

# ============================================================================
# BLOCK 4: TF-IDF ANALYSIS AND KEYWORD DISCOVERY
# ============================================================================

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,
    min_df=5,  # Ignore terms that appear in less than 5 documents
    max_df=0.7,  # Ignore terms that appear in more than 70% of documents
    ngram_range=(1, 2),  # Include unigrams and bigrams
    stop_words='english'
)

# Fit TF-IDF on cleaned text
print("🔄 Computing TF-IDF vectors...")
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
feature_names = tfidf.get_feature_names_out()

print(f"✅ TF-IDF completed! Matrix shape: {tfidf_matrix.shape}")

# Get top keywords overall
def get_top_keywords(tfidf_matrix, feature_names, top_n=20):
    """Get top keywords by TF-IDF score"""
    # Sum TF-IDF scores across all documents
    scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    
    # Create keyword-score pairs and sort
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]

top_keywords = get_top_keywords(tfidf_matrix, feature_names, 30)

print(f"\n🔝 Top 30 keywords by TF-IDF importance:")
for i, (keyword, score) in enumerate(top_keywords, 1):
    print(f"{i:2d}. {keyword:<20} (score: {score:.3f})")

# ============================================================================
# BLOCK 5: SEMANTIC CLUSTERING FOR AUTOMATIC CATEGORY DISCOVERY
# ============================================================================

# Use K-means clustering to discover natural groupings
print("\n🔄 Discovering natural categories using K-means clustering...")

# Try different numbers of clusters
cluster_range = range(3, 8)
inertias = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Use optimal number of clusters (let's start with 5 based on user's categories)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

df['auto_cluster'] = cluster_labels

print(f"✅ Clustering completed with {optimal_clusters} clusters")
print(f"📊 Cluster distribution:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"Cluster {cluster_id}: {count:,} records ({percentage:.1f}%)")

# ============================================================================
# BLOCK 6: ANALYZE CLUSTERS AND EXTRACT THEMES
# ============================================================================

def analyze_cluster_themes(cluster_id, top_n=15):
    """Analyze what each cluster represents"""
    cluster_mask = df['auto_cluster'] == cluster_id
    cluster_docs = df[cluster_mask]['cleaned_text'].values
    
    if len(cluster_docs) == 0:
        return []
    
    # Get TF-IDF for this cluster
    cluster_tfidf = tfidf.transform(cluster_docs)
    
    # Calculate mean TF-IDF scores for this cluster
    mean_scores = np.array(cluster_tfidf.mean(axis=0)).flatten()
    
    # Get top keywords for this cluster
    keyword_scores = list(zip(feature_names, mean_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]

print("\n🎯 CLUSTER THEMES ANALYSIS:")
print("="*60)

cluster_themes = {}
for cluster_id in range(optimal_clusters):
    themes = analyze_cluster_themes(cluster_id)
    cluster_themes[cluster_id] = themes
    
    print(f"\n📂 CLUSTER {cluster_id} (n={cluster_counts[cluster_id]:,}):")
    print("Top keywords:", ", ".join([kw for kw, score in themes[:10]]))
    
    # Show sample descriptions
    cluster_samples = df[df['auto_cluster'] == cluster_id]['AccidentDesc'].head(3)
    print("Sample descriptions:")
    for i, desc in enumerate(cluster_samples, 1):
        print(f"  {i}. {desc[:100]}...")

# ============================================================================
# BLOCK 7: SMART CATEGORY MAPPING USING MEDICAL NER AND SEMANTIC ANALYSIS
# ============================================================================

def classify_using_semantic_analysis(text, medical_entities):
    """Classify using semantic analysis and medical entities"""
    text_lower = text.lower()
    
    # Define semantic keyword groups (more comprehensive than manual keywords)
    illness_indicators = {
        'symptoms': ['fever', 'cough', 'headache', 'pain', 'ache', 'sick', 'ill', 'flu', 'cold'],
        'conditions': ['allergic', 'reaction', 'infection', 'virus', 'bacterial', 'respiratory'],
        'treatments': ['medication', 'treatment', 'hospital', 'emergency', 'medical']
    }
    
    accident_indicators = {
        'injuries': ['fracture', 'broken', 'sprain', 'bruise', 'cut', 'wound', 'injury', 'hurt'],
        'body_parts': ['knee', 'shoulder', 'back', 'neck', 'ankle', 'arm', 'leg', 'head', 'ribs'],
        'incidents': ['accident', 'collision', 'fall', 'fell', 'slip', 'crash', 'hit', 'impact']
    }
    
    food_indicators = {
        'poisoning': ['food poisoning', 'stomach', 'nausea', 'vomit', 'diarrhea', 'diarrhoea'],
        'digestive': ['gastro', 'intestinal', 'digestive', 'bowel', 'abdominal'],
        'food_types': ['shellfish', 'seafood', 'restaurant', 'cafe', 'meal']
    }
    
    # Calculate semantic scores
    illness_score = sum([
        sum(1 for keyword in keywords if keyword in text_lower)
        for keywords in illness_indicators.values()
    ])
    
    accident_score = sum([
        sum(1 for keyword in keywords if keyword in text_lower)
        for keywords in accident_indicators.values()
    ])
    
    food_score = sum([
        sum(1 for keyword in keywords if keyword in text_lower)
        for keywords in food_indicators.values()
    ])
    
    # Boost scores based on medical entities
    medical_boost = len(medical_entities) * 0.5
    if any(entity in ['stomach', 'digestive', 'intestinal'] for entity in medical_entities):
        food_score += medical_boost
    elif any(entity in ['fever', 'headache', 'respiratory'] for entity in medical_entities):
        illness_score += medical_boost
    elif any(entity in ['injury', 'fracture', 'sprain'] for entity in medical_entities):
        accident_score += medical_boost
    
    return {
        'general_illness': illness_score > 0,
        'travel_accident': accident_score > 0,
        'food_poisoning': food_score > 0,
        'scores': {
            'illness_score': illness_score,
            'accident_score': accident_score,
            'food_score': food_score
        }
    }

print("🔄 Applying semantic classification...")

# Apply semantic classification
semantic_results = []
for idx, row in df.iterrows():
    result = classify_using_semantic_analysis(row['cleaned_text'], row['medical_entities'])
    semantic_results.append(result)

# Extract classification results
df['general_illness_flag'] = [result['general_illness'] for result in semantic_results]
df['travel_accident_flag'] = [result['travel_accident'] for result in semantic_results]
df['food_poisoning_flag'] = [result['food_poisoning'] for result in semantic_results]

# Calculate category statistics
category_columns = ['general_illness_flag', 'travel_accident_flag', 'food_poisoning_flag']
df['category_count'] = df[category_columns].sum(axis=1)

# Create category labels
def get_category_labels(row):
    categories = []
    if row['general_illness_flag']:
        categories.append('General Illness')
    if row['travel_accident_flag']:
        categories.append('Travel/Accident')
    if row['food_poisoning_flag']:
        categories.append('Food Poisoning')
    
    return ', '.join(categories) if categories else 'Uncategorized'

df['semantic_categories'] = df.apply(get_category_labels, axis=1)

print("✅ Semantic classification completed!")

# ============================================================================
# BLOCK 8: RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================

# Generate comprehensive summary
print("\n📊 SEMANTIC CLASSIFICATION RESULTS:")
print("="*50)

total_records = len(df)
general_illness_count = df['general_illness_flag'].sum()
travel_accident_count = df['travel_accident_flag'].sum()
food_poisoning_count = df['food_poisoning_flag'].sum()
multiple_categories = len(df[df['category_count'] > 1])
uncategorized = len(df[df['category_count'] == 0])
classification_rate = ((total_records - uncategorized) / total_records) * 100

print(f"Total Records: {total_records:,}")
print(f"Classification Rate: {classification_rate:.1f}%")
print(f"Uncategorized: {uncategorized:,} ({(uncategorized/total_records)*100:.1f}%)")
print(f"Multiple Categories: {multiple_categories:,} ({(multiple_categories/total_records)*100:.1f}%)")

print(f"\nCATEGORY BREAKDOWN:")
print(f"General Illness: {general_illness_count:,} ({(general_illness_count/total_records)*100:.1f}%)")
print(f"Travel/Accident: {travel_accident_count:,} ({(travel_accident_count/total_records)*100:.1f}%)")
print(f"Food Poisoning: {food_poisoning_count:,} ({(food_poisoning_count/total_records)*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Category distribution
category_counts = [general_illness_count, travel_accident_count, food_poisoning_count, uncategorized]
category_labels = ['General Illness', 'Travel/Accident', 'Food Poisoning', 'Uncategorized']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

axes[0, 0].pie(category_counts, labels=category_labels, autopct='%1.1f%%', colors=colors)
axes[0, 0].set_title('Category Distribution')

# Cluster vs Semantic Category comparison
cluster_category_crosstab = pd.crosstab(df['auto_cluster'], df['semantic_categories'])
sns.heatmap(cluster_category_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Auto Clusters vs Semantic Categories')
axes[0, 1].set_xlabel('Semantic Categories')
axes[0, 1].set_ylabel('Auto Clusters')

# Description length by category
category_lengths = df.groupby('semantic_categories')['desc_length'].mean()
axes[1, 0].bar(range(len(category_lengths)), category_lengths.values)
axes[1, 0].set_xticks(range(len(category_lengths)))
axes[1, 0].set_xticklabels(category_lengths.index, rotation=45)
axes[1, 0].set_title('Average Description Length by Category')
axes[1, 0].set_ylabel('Characters')

# Multiple categories analysis
multi_cat_df = df[df['category_count'] > 1]
if len(multi_cat_df) > 0:
    multi_cat_counts = multi_cat_df['semantic_categories'].value_counts()
    axes[1, 1].bar(range(len(multi_cat_counts)), multi_cat_counts.values)
    axes[1, 1].set_xticks(range(len(multi_cat_counts)))
    axes[1, 1].set_xticklabels(multi_cat_counts.index, rotation=45)
    axes[1, 1].set_title('Multiple Category Combinations')
    axes[1, 1].set_ylabel('Count')
else:
    axes[1, 1].text(0.5, 0.5, 'No multiple\ncategories found', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Multiple Category Combinations')

plt.tight_layout()
plt.show()

# ============================================================================
# BLOCK 9: SAVE RESULTS AND CREATE CATEGORY-SPECIFIC DATASETS
# ============================================================================

# Save main classified dataset
output_file = '/Users/bervynwong/Desktop/INCOME Travel Insurance Portfolio Analysis Project/Profitability Model/Free Text classification/accident_description_classified.csv'
df.to_csv(output_file, index=False)
print(f"💾 Main classified dataset saved to: {output_file}")

# Create category-specific datasets
categories = {
    'general_illness': df[df['general_illness_flag'] == True],
    'travel_accident': df[df['travel_accident_flag'] == True],
    'food_poisoning': df[df['food_poisoning_flag'] == True],
    'multiple_categories': df[df['category_count'] > 1],
    'uncategorized': df[df['category_count'] == 0]
}

print(f"\n💾 Creating category-specific datasets:")
for category_name, category_df in categories.items():
    if len(category_df) > 0:
        filename = f"{category_name}_descriptions_nlp.csv"
        category_df.to_csv(filename, index=False)
        print(f"✅ {category_name.replace('_', ' ').title()}: {len(category_df):,} records → {filename}")

# ============================================================================
# BLOCK 10: SAMPLE CLASSIFICATIONS AND VALIDATION
# ============================================================================

print(f"\n🔍 SAMPLE CLASSIFICATIONS:")
print("="*80)

# Show examples from each category
for category in ['General Illness', 'Travel/Accident', 'Food Poisoning']:
    category_samples = df[df['semantic_categories'] == category].head(3)
    if len(category_samples) > 0:
        print(f"\n📂 {category.upper()} SAMPLES:")
        for idx, row in category_samples.iterrows():
            print(f"Description: {row['AccidentDesc']}")
            print(f"Cleaned: {row['cleaned_text'][:100]}...")
            print(f"Medical entities: {row['medical_entities']}")
            print(f"Auto cluster: {row['auto_cluster']}")
            print("-" * 50)

# Show multiple category examples
multi_samples = df[df['category_count'] > 1].head(3)
if len(multi_samples) > 0:
    print(f"\n📂 MULTIPLE CATEGORIES SAMPLES:")
    for idx, row in multi_samples.iterrows():
        print(f"Description: {row['AccidentDesc']}")
        print(f"Categories: {row['semantic_categories']}")
        print(f"Flags: Illness={row['general_illness_flag']}, Accident={row['travel_accident_flag']}, Food={row['food_poisoning_flag']}")
        print("-" * 50)

print(f"\n🎉 NLP Classification completed successfully!")
print(f"📈 Total processing: {len(df):,} records classified with {classification_rate:.1f}% success rate")
print(f"📁 Output files created: {output_file} + category-specific CSV files")