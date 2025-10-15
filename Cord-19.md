# CORD-19 Data Explorer: Analysis and Streamlit Application Report

**Author: Fortune Akioya**

## 1. Project Overview

This project aims to perform a comprehensive data analysis on the `metadata.csv` file from the CORD-19 dataset. The CORD-19 (COVID-19 Open Research Dataset) is a collection of scholarly articles about COVID-19 and related historical coronavirus research. The analysis covers data loading, exploration, cleaning, statistical summarization, and visualization of key trends, culminating in an interactive Streamlit web application.

## 2. Methodology

The project was structured into several parts, following a typical data science workflow:

### Part 1: Data Loading and Basic Exploration
*   The `metadata.csv` file was loaded into a Pandas DataFrame.
*   Initial inspection involved `df.head()`, `df.info()`, `df.shape`, and `df.isnull().sum()` to understand the data's structure, data types, and presence of missing values.
*   `df.describe()` provided statistical summaries for numerical columns.

### Part 2: Data Cleaning and Preparation
*   **Missing Values:**
    *   Rows with missing `publish_time` were dropped as this column is crucial for time-series analysis.
    *   Missing `title` and `abstract` values were filled with 'No Title' and 'No Abstract' respectively to retain paper entries that might still have other valuable metadata.
    *   Missing `journal` entries were filled with 'Unknown Journal'.
*   **Data Type Conversion:** The `publish_time` column was converted to datetime objects, with `errors='coerce'` to handle malformed dates gracefully, followed by dropping rows where conversion failed.
*   **Feature Engineering:**
    *   `publication_year` was extracted from `publish_time` for time-based analysis. Unrealistic years (e.g., pre-1900 or future dates) were filtered out.
    *   `abstract_word_count` was created to quantify the length of abstracts.
*   **Deduplication:** Duplicate papers (based on identical title and abstract) were removed to ensure unique entries.

### Part 3: Data Analysis and Visualization
*   **Temporal Trends:** Count of papers by publication year was computed to visualize research output over time.
*   **Top Journals:** Identified the journals that published the most COVID-19 related research.
*   **Frequent Words:** A simple word frequency analysis was performed on paper titles to identify common themes and keywords.
*   **Visualizations:**
    *   **Publications Over Time:** A bar chart showing the number of research papers published each year.
    *   **Top Publishing Journals:** A bar chart displaying the top N journals by publication count.
    *   **Word Cloud of Paper Titles:** A visual representation of frequently occurring words in titles, emphasizing key research areas.
    *   **Distribution of Paper Counts by Source:** A bar chart showing the number of papers originating from different sources (e.g., PMC, CZI).

### Part 4: Streamlit Application
*   An interactive web application was built using Streamlit, allowing users to explore the cleaned data and visualizations.
*   Interactive widgets like sliders for year range, number of top journals, and word cloud size were implemented.
*   The app displays raw data exploration, cleaning summary (in sidebar), cleaned data overview, all generated visualizations, and a sample of the cleaned data.

## 3. Findings and Observations

1.  **Explosive Growth in 2020-2021:** The plot of publications over time clearly shows a massive surge in COVID-19 related research starting in late 2019/early 2020, peaking around 2020-2021, reflecting the global response to the pandemic.
2.  **Dominant Journals:** A few key journals and pre-print servers tend to publish a significant volume of research in this area, indicating central hubs for dissemination.
3.  **Key Research Themes:** Word clouds and frequent word lists from titles highlight recurring terms like "patient," "health," "public," "infection," "virus," "treatment," and "vaccine," which are central to COVID-19 research.
4.  **Source Diversity:** The dataset integrates papers from various sources (e.g., PubMed Central, bioRxiv, medRxiv), demonstrating the broad compilation of CORD-19.

## 4. Challenges and Learning

*   **Handling Large Datasets:** The CORD-19 metadata can be quite large. Efficient data loading and caching (using `@st.cache_data` in Streamlit) were crucial for performance.
*   **Inconsistent Date Formats:** The `publish_time` column often contains various date formats. Using `pd.to_datetime(..., errors='coerce')` proved effective for robust conversion, although it required subsequent dropping of unparseable rows.
*   **Text Cleaning for Word Clouds:** Simple word frequency and word cloud generation required basic text cleaning (removing punctuation, common stop words, and very short words) to produce meaningful insights from titles and abstracts.
*   **Streamlit Development:** Learning to integrate Matplotlib/Seaborn plots into Streamlit, using interactive widgets (`st.slider`), and organizing the application layout (`st.sidebar`, `st.header`) was a valuable experience.
*   **Iterative Cleaning:** Data cleaning is not a one-step process. Observing the effects of initial cleaning steps often reveals further inconsistencies that need addressing (e.g., filtering out unrealistic years).

## 5. Conclusion

This project successfully demonstrates a full data science workflow from raw data to an interactive application. It provides valuable insights into the publication landscape of COVID-19 research, showcasing the power of Python libraries like Pandas, Matplotlib, Seaborn, and Streamlit for rapid data exploration and communication. The interactive Streamlit app makes the findings accessible and explorable for a wider audience.
