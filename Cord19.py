# Author: Fortune Akioya
# Project: CORD-19 Metadata Analysis and Streamlit App

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from collections import Counter
from wordcloud import WordCloud

# --- Part 1: Download and Load the Data & Basic Data Exploration ---

@st.cache_data # Cache the data loading for efficiency in Streamlit
def load_data(file_path='metadata.csv'):
    """
    Loads the metadata.csv file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please download the metadata.csv "
                 "from the CORD-19 dataset and place it in the same directory as this script.")
        st.stop() # Stop the Streamlit app if data isn't found
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# --- Part 2: Data Cleaning and Preparation ---

@st.cache_data # Cache the cleaning process
def clean_and_prepare_data(df):
    """
    Handles missing values, converts date columns, and creates new features.
    """
    # Create a copy to avoid modifying the original DataFrame directly
    cleaned_df = df.copy()

    st.sidebar.markdown("### Data Cleaning & Preparation Summary")

    # Check for missing values in important columns
    initial_missing = cleaned_df[['title', 'abstract', 'publish_time', 'journal']].isnull().sum()
    st.sidebar.write("Initial missing values in key columns:")
    st.sidebar.write(initial_missing)

    # Handle missing 'publish_time': Drop rows where publish_time is missing as it's crucial for time analysis
    original_rows = len(cleaned_df)
    cleaned_df.dropna(subset=['publish_time'], inplace=True)
    st.sidebar.write(f"- Dropped {original_rows - len(cleaned_df)} rows with missing 'publish_time'.")

    # Handle missing 'title' and 'abstract': Fill with 'No Title'/'No Abstract'
    cleaned_df['title'].fillna('No Title', inplace=True)
    cleaned_df['abstract'].fillna('No Abstract', inplace=True)
    st.sidebar.write("- Filled missing 'title' with 'No Title' and 'abstract' with 'No Abstract'.")

    # Handle missing 'journal': Fill with 'Unknown Journal'
    cleaned_df['journal'].fillna('Unknown Journal', inplace=True)
    st.sidebar.write("- Filled missing 'journal' with 'Unknown Journal'.")


    # Convert 'publish_time' to datetime format
    # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
    cleaned_df['publish_time'] = pd.to_datetime(cleaned_df['publish_time'], errors='coerce')

    # Drop rows where 'publish_time' became NaT after coercion
    original_rows_after_publish_time_conversion = len(cleaned_df)
    cleaned_df.dropna(subset=['publish_time'], inplace=True)
    st.sidebar.write(f"- Dropped {original_rows_after_publish_time_conversion - len(cleaned_df)} rows with invalid 'publish_time' format.")


    # Extract year from publication date
    cleaned_df['publication_year'] = cleaned_df['publish_time'].dt.year

    # Filter out unreasonable years (e.g., year 0 or future dates)
    cleaned_df = cleaned_df[cleaned_df['publication_year'] > 1900] # Assuming CORD-19 data is modern research
    cleaned_df = cleaned_df[cleaned_df['publication_year'] <= pd.Timestamp.now().year]
    st.sidebar.write("- Extracted 'publication_year' and filtered out unrealistic years.")


    # Create 'abstract_word_count'
    cleaned_df['abstract_word_count'] = cleaned_df['abstract'].apply(lambda x: len(str(x).split()))
    st.sidebar.write("- Created 'abstract_word_count' column.")

    # Remove duplicates based on 'title' and 'abstract'
    original_rows_before_deduplication = len(cleaned_df)
    cleaned_df.drop_duplicates(subset=['title', 'abstract'], inplace=True)
    st.sidebar.write(f"- Removed {original_rows_before_deduplication - len(cleaned_df)} duplicate papers based on title and abstract.")


    final_missing = cleaned_df[['title', 'abstract', 'publish_time', 'journal', 'publication_year']].isnull().sum()
    st.sidebar.write("\nFinal missing values in key columns:")
    st.sidebar.write(final_missing)
    st.sidebar.success("Data cleaning and preparation complete!")

    return cleaned_df

# --- Part 3: Data Analysis and Visualization Functions ---

def plot_publications_over_time(df, year_range):
    """Plots the number of publications over time."""
    df_filtered = df[(df['publication_year'] >= year_range[0]) & (df['publication_year'] <= year_range[1])]
    year_counts = df_filtered['publication_year'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_counts.index, year_counts.values, color='skyblue')
    ax.set_title(f'Number of Publications by Year ({year_range[0]}-{year_range[1]})')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Papers')
    ax.set_xticks(year_counts.index)
    ax.tick_params(axis='x', rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_top_journals(df, top_n=10):
    """Plots a bar chart of top publishing journals."""
    journal_counts = df['journal'].value_counts().nlargest(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=journal_counts.values, y=journal_counts.index, palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Journals Publishing CORD-19 Research')
    ax.set_xlabel('Number of Papers')
    ax.set_ylabel('Journal')
    plt.tight_layout()
    return fig

def generate_word_cloud(df, column='title', max_words=100):
    """Generates a word cloud from a specified text column."""
    # Filter out 'No Title' or 'No Abstract'
    text = " ".join(df[df[column] != f'No {column.capitalize()}'][column].dropna().tolist())

    # Basic cleaning: remove common words and single characters
    # More sophisticated stop word removal could be done here
    stop_words = set(WordCloud().stopwords)
    stop_words.update(['no', 'abstract', 'title', 'covid', 'coronavirus', 'sars', 'cov', 'report', 'study', 'case'])
    text = re.sub(r'\b\w{1,2}\b', '', text) # Remove short words
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words, max_words=max_words, collocations=False).generate(text.lower())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud of Paper {column.capitalize()}s')
    plt.tight_layout()
    return fig

def plot_source_distribution(df, top_n=10):
    """Plots the distribution of paper counts by source."""
    source_counts = df['source_x'].value_counts().nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=source_counts.index, y=source_counts.values, palette='plasma', ax=ax)
    ax.set_title(f'Top {top_n} Sources of CORD-19 Papers')
    ax.set_xlabel('Source')
    ax.set_ylabel('Number of Papers')
    ax.tick_params(axis='x', rotation=45, ha='right')
    plt.tight_layout()
    return fig

def get_most_frequent_words_in_titles(df, top_n=20):
    """Finds the most frequent words in paper titles."""
    titles = df[df['title'] != 'No Title']['title'].dropna().str.lower().tolist()
    all_words = []
    for title in titles:
        # Basic cleaning: remove punctuation and split
        words = re.findall(r'\b\w+\b', title)
        all_words.extend(words)

    # Filter out common stop words and very short words
    stop_words = set(WordCloud().stopwords)
    stop_words.update(['no', 'title', 'covid', 'coronavirus', 'sars', 'cov', 'report', 'study', 'case', 'review', 'analysis', 'impact', 'effect'])
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]

    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

# --- Part 4: Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="CORD-19 Data Explorer", page_icon="🔬")

    st.title("CORD-19 Data Explorer 🔬")
    st.markdown("### Simple Exploration of COVID-19 Research Papers")
    st.write("This application allows you to explore the metadata of the CORD-19 dataset, "
             "analyzing publication trends, top journals, and frequently discussed topics.")

    st.sidebar.header("Configuration")
    file_path = st.sidebar.text_input("Path to metadata.csv", "metadata.csv")

    df = load_data(file_path)

    st.info(f"Loaded {len(df)} rows and {df.shape[1]} columns from '{file_path}'.")

    # --- Basic Data Exploration Display ---
    st.header("1. Basic Data Exploration")
    st.subheader("DataFrame Dimensions")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("First 5 Rows of the Dataset")
    st.dataframe(df.head())

    st.subheader("Data Types of Each Column")
    st.write(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

    st.subheader("Missing Values (Raw Data)")
    st.dataframe(df.isnull().sum().to_frame(name='Missing Count').loc[['title', 'abstract', 'publish_time', 'journal', 'source_x']])

    st.subheader("Basic Statistics for Numerical Columns (Raw Data)")
    st.dataframe(df.describe())

    st.markdown("---")

    # Clean and prepare the data
    st.sidebar.header("Data Cleaning Progress")
    cleaned_df = clean_and_prepare_data(df)

    st.header("2. Cleaned Data Overview")
    st.write(f"After cleaning, the dataset contains {len(cleaned_df)} rows and {cleaned_df.shape[1]} columns.")
    st.subheader("First 5 Rows of Cleaned Data")
    st.dataframe(cleaned_df.head())

    st.subheader("Basic Statistics for Numerical Columns (Cleaned Data)")
    st.dataframe(cleaned_df.describe())

    st.markdown("---")

    # --- Part 3: Data Analysis and Visualization Display ---
    st.header("3. Data Analysis and Visualizations")

    # Interactive elements for year range
    min_year = int(cleaned_df['publication_year'].min())
    max_year = int(cleaned_df['publication_year'].max())
    year_range_selection = st.sidebar.slider(
        "Select Publication Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, 2019), max_year) # Default to 2019-max_year for CORD-19 relevance
    )

    st.subheader("Publications Over Time")
    st.pyplot(plot_publications_over_time(cleaned_df, year_range_selection))
    ` `
    st.subheader("Top Publishing Journals")
    top_journals_n = st.sidebar.slider("Number of Top Journals", 5, 20, 10)
    st.pyplot(plot_top_journals(cleaned_df, top_journals_n))
    ` `
    st.subheader("Word Cloud of Paper Titles")
    wordcloud_max_words = st.sidebar.slider("Max Words in Word Cloud", 50, 200, 100)
    st.pyplot(generate_word_cloud(cleaned_df, column='title', max_words=wordcloud_max_words))
    ` `
    st.subheader("Distribution of Paper Counts by Source")
    top_sources_n = st.sidebar.slider("Number of Top Sources", 3, 15, 7)
    st.pyplot(plot_source_distribution(cleaned_df, top_sources_n))
    ` `
    st.subheader("Most Frequent Words in Titles")
    most_freq_words_n = st.sidebar.slider("Number of Most Frequent Words", 10, 50, 20)
    frequent_words = get_most_frequent_words_in_titles(cleaned_df, most_freq_words_n)
    st.write(pd.DataFrame(frequent_words, columns=['Word', 'Frequency']))

    st.markdown("---")

    st.header("4. Sample of Cleaned Data")
    st.dataframe(cleaned_df.sample(min(10, len(cleaned_df)))) # Display max 10 random rows

    st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About")
    st.sidebar.info("This app explores the CORD-19 dataset metadata to visualize research trends.")


if __name__ == "__main__":
    main()
