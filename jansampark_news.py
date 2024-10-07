from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def initialize_webdriver():
    """Initialize a headless Selenium Chrome WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


def fetch_articles(url):
    """Fetch articles from the provided URL using Selenium."""
    driver = initialize_webdriver()
    try:
        driver.get(url)

        # Scroll down to load more content
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'IFHyqb'))
        )
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup.find_all('article', class_='IFHyqb')
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []
    finally:
        driver.quit()


def extract_article_data(articles):
    """Extracts article data such as title, link, source, and datetime."""
    data = []
    titles_seen = set()

    for article in articles:
        try:
            title_tag = article.find('a', class_='JtKRv')
            title = title_tag.get_text(strip=True) if title_tag else 'N/A'
            link = f"https://news.google.com{title_tag['href']}" if title_tag else 'N/A'
            source_tag = article.find('div', class_='vr1PYe')
            source = source_tag.get_text(strip=True) if source_tag else 'N/A'
            time_tag = article.find('time', class_='hvbAAd')
            date_time = time_tag.get_text(strip=True) if time_tag else 'N/A'

            # Avoid duplicate titles
            if title.lower().strip() not in titles_seen:
                titles_seen.add(title.lower().strip())
                data.append({
                    'Title': title,
                    'Link': link,
                    'Source': source,
                    'DateTime': date_time,
                    'OriginalDateTime': date_time
                })

        except Exception as e:
            print(f"Error processing article: {e}")
    
    return data


def compute_unique_titles(data, model):
    """Computes unique articles using sentence embeddings."""
    titles = [entry['Title'] for entry in data]
    embeddings = model.encode(titles, convert_to_tensor=True)

    cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    similarity_threshold = 0.7
    unique_indices = []
    processed = np.zeros(len(titles), dtype=bool)

    for i in range(len(titles)):
        if not processed[i]:
            unique_indices.append(i)
            for j in range(i + 1, len(titles)):
                if cosine_sim_matrix[i][j] > similarity_threshold:
                    processed[j] = True

    unique_data = [data[i] for i in unique_indices]
    return unique_data


def save_to_csv(data, filename):
    """Saves the data to a CSV file."""
    df_new = pd.DataFrame(data)
    print(f"New DataFrame shape: {df_new.shape}")

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        print(f"Existing DataFrame shape: {df_existing.shape}")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"Combined DataFrame shape before dropping duplicates: {df_combined.shape}")
        df_combined.drop_duplicates(subset=['Title'], keep='first', inplace=True)
        print(f"Combined DataFrame shape after dropping duplicates: {df_combined.shape}")
    else:
        df_combined = df_new
    
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Unique data has been saved to '{filename}' with {df_combined.shape[0]} records.")


def process_date(date_str):
    time_units = {
        'second[s]?': 'A',
        'minute[s]?': 'B',
        'hour[s]?': 'C',
        'day[s]?': 'D',
        'Yesterday': 'D',
        'yesterday': 'D',
        'week[s]?': 'E',
        'month[s]?': 'F',
        'year[s]?': 'G'
    }

    for unit, group in time_units.items():
        match = re.search(rf'(\d+)?\s*{unit}|yesterday', date_str, re.IGNORECASE)
        if match:
            time_value = match.group(1)

            if 'yesterday' in date_str.lower():
                return 'D1 Yesterday'
            
            if time_value:
                time_value = int(time_value)
                
                # Define numeric ranks for each group
                if group == 'A':  # Seconds
                    if time_value <= 9:
                        rank = 'A1'
                    elif time_value <= 19:
                        rank = 'A2'
                    else:
                        rank = 'A3'
                
                elif group == 'B':  # Minutes
                    if time_value <= 9:
                        rank = 'B1'
                    elif time_value <= 19:
                        rank = 'B2'
                    elif time_value <= 29:
                        rank = 'B3'
                    else:
                        rank = 'B4'

                elif group == 'C':  # Hours
                    if time_value <= 9:
                        rank = 'C1'
                    elif time_value <= 19:
                        rank = 'C2'
                    elif time_value <= 29:
                        rank = 'C3'
                    elif time_value <= 39:
                        rank = 'C4'
                    else:
                        rank = 'C5'
                
                elif group == 'D':  # Days
                    if time_value <= 9:
                        rank = 'D1'
                    elif time_value <= 19:
                        rank = 'D2'
                    elif time_value <= 29:
                        rank = 'D3'
                    elif time_value <= 39:
                        rank = 'D4'
                    else:
                        rank = 'D5'
                
                elif group == 'E':  # Weeks
                    if time_value <= 9:
                        rank = 'E1'
                    elif time_value <= 19:
                        rank = 'E2'
                    elif time_value <= 29:
                        rank = 'E3'
                    else:
                        rank = 'E4'

                elif group == 'F':  # Months
                    if time_value <= 9:
                        rank = 'F1'
                    elif time_value <= 19:
                        rank = 'F2'
                    elif time_value <= 29:
                        rank = 'F3'
                    else:
                        rank = 'F4'
                
                elif group == 'G':  # Years
                    if time_value <= 9:
                        rank = 'G1'
                    elif time_value <= 19:
                        rank = 'G2'
                    elif time_value <= 29:
                        rank = 'G3'
                    else:
                        rank = 'G4'

                return f"{rank} {time_value} {unit}"
    
    return date_str


def is_similar_article(title, all_articles, model, threshold=0.85):
    """Checks if an article is similar to any existing articles."""
    combined_text = title
    combined_embeddings = model.encode([combined_text], convert_to_tensor=True)
    
    with torch.no_grad():
        if all_articles:
            article_embeddings = model.encode(all_articles, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(combined_embeddings, article_embeddings)
            return torch.max(cosine_scores).item() > threshold
    return False

from transformers import pipeline

# Load a model that can classify sentiment into multiple categories
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def use_gen_ai_for_sentiment(text):
    """Use a transformer model to perform sentiment analysis with three categories: Positive, Negative, Sensitive."""
    
    # Analyze the sentiment using the pipeline
    results = sentiment_pipeline(text)
    
    # Extract the sentiment label and score
    sentiment = results[0]['label']
    score = results[0]['score']
    
    # Map the model's output to your desired categories
    if "Positive" in sentiment:
        return "Positive"
    elif "Negative" in sentiment:
        return "Negative"
    else:
        return "Sensitive"  # If the model predicts any other category

def process_existing_data(unique_data, model):
    """Processes unique data for sentiment and categorization."""
    all_articles = []
    processed_data = []

    for entry in unique_data:
        try:
            title = entry['Title'].strip()
            date = process_date(entry['DateTime'])
            original_date = entry['OriginalDateTime']
            source = entry['Source']
            link = entry['Link']

            sentiment = use_gen_ai_for_sentiment(title)

            # Check if the article is similar to an existing one
            if not is_similar_article(title, all_articles, model):
                all_articles.append(title)
                processed_data.append({
                    'Title': title,
                    'Link': link,
                    'Source': source,
                    'Date': original_date,
                    'Sentiment': sentiment
                })

        except Exception as e:
            print(f"Error in processing: {e}")

    return processed_data


def main():
    url = 'https://news.google.com/search?q=Madhya%20Pradesh%20government%20latest%20news&hl=en-IN&gl=IN&ceid=IN%3Ahi'
    articles = fetch_articles(url)

    if articles:
        article_data = extract_article_data(articles)
        unique_data = compute_unique_titles(article_data, model)
        processed_data = process_existing_data(unique_data, model)
        save_to_csv(processed_data, 'articles.csv')
    else:
        print("No articles found.")


if __name__ == "__main__":
    main()
