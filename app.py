import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
import base64

# Streamlit UI
st.set_page_config(page_title="News Analyzer", page_icon="📰", layout="wide")

# Function to fetch articles (simulate for now)
def fetch_articles():
    # Initialize a placeholder for the loading message
    loading_message = st.empty()
    loading_message.info("Fetching latest news articles from the existing CSV file...")

    # Path to your existing articles.csv file
    csv_file_path = 'articles.csv'

    try:
        # Read the CSV file into a pandas DataFrame
        articles_df = pd.read_csv(csv_file_path)
        # After successful loading, clear the loading message
        loading_message.empty()
        st.success(f"Successfully loaded {len(articles_df)} articles.")
        return articles_df

    except FileNotFoundError:
        # Clear the loading message if file not found
        loading_message.empty()
        st.error("Error: 'articles.csv' file not found. Please make sure the file exists.")
        return pd.DataFrame()

    except Exception as e:
        # Clear the loading message if there's any other exception
        loading_message.empty()
        st.error(f"An error occurred while loading the CSV file: {e}")
        return pd.DataFrame()
    
# Function to make file links clickable in the DataFrame
def make_clickable(val):
    return f'<a href="{val}" target="_blank">{val[:30]}...</a>'  # Truncate long URLs for better display

# Display the logo without resizing
with st.container():
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2], gap="small")
    
    with col1:
        st.markdown('<div style="display: flex; align-items: flex-end; justify-content: bottom; height: 180%;">', unsafe_allow_html=True)
        logo_image = Image.open('image/jansampark-ribbon.jfif')
        st.image(logo_image, use_column_width='auto')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown("# 📰 JanSampark News Analyzer")
        st.markdown("###### AI Based News Analyzer")
    
    with col3:
        l = Image.open('image/mpsedc-logo.png')
        re = l.resize((165, 127))  # Corrected the resize method call
        st.image(re)


# Function to add background image from a local file
def add_bg_from_local(image_file, opacity=0):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

    # Inject custom CSS for background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url(data:image/jfif;base64,{encoded_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Call the function with your image path
add_bg_from_local('image/grey_bg.jfif')  

st.markdown(
    """
    <style>
        .footer {
            position: absolute;
            top: 80px;
            left: 0;
            width: 100%;
            background-color: #002F74;
            color: white;
            text-align: center;
            padding: 5px;
            font-weight: bold;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .footer p {
            font-style: italic;
            font-size: 14px;
            margin: 0;
            flex: 1 1 50%;
        }
        .content {
            padding: 10px;
            margin-top: 20px;
        }
        .title {
            margin-bottom: 30px;
            word-wrap: break-word;
        }
        .dataframe td {
            max-width: 600px;
            white-space: normal;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .dataframe th {
            text-align: left;
        }
        .dataframe tr:hover {
            background-color: #f1f1f1;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""<div class="title">Welcome to the MP News Analyzer app! This tool fetches the latest news articles related to Madhya Pradesh and performs sentiment analysis, translation, and categorization.</div>""", unsafe_allow_html=True)
st.markdown("""You can view the fetched articles, filter them based on sentiment or date, and get insights into the overall sentiment distribution.""")

# Main section for filters
st.header("Filter Articles")
sentiment_filter = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative","Sensitive"], default=["Positive", "Neutral", "Negative","Sensitive"])

# Button to fetch articles
if st.button("Fetch Latest News"):
    articles_df = fetch_articles()


    # Count total articles and by sentiment
    total_count = len(articles_df)
    sentiment_counts = articles_df['Sentiment'].value_counts().to_dict()

    # Fill in missing sentiment categories with 0
    for sentiment in ["Positive", "Neutral", "Negative", "Sensitive"]:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Sentiment': ['Total', 'Positive', 'Neutral', 'Negative', 'Sensitive'],
        'Count': [total_count, sentiment_counts['Positive'], sentiment_counts['Neutral'], sentiment_counts['Negative'], sentiment_counts['Sensitive']]
    })

    # Apply filters
    filtered_df = articles_df[articles_df['Sentiment'].isin(sentiment_filter)]

    # Make the 'Link' column clickable
    if 'Link' in filtered_df.columns:
        filtered_df['Link'] = filtered_df['Link'].apply(make_clickable)

    # Display the articles in a table format with clickable links
    st.subheader("Fetched Articles")
    st.write(f"Showing {len(filtered_df)} articles matching the sentiment filter.")

    # Display the filtered DataFrame with clickable links
    st.markdown(
        filtered_df.to_html(escape=False, index=False), 
        unsafe_allow_html=True
    )

    # Button to download the filtered articles data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download data as CSV", data=csv, file_name="filtered_articles.csv", mime='text/csv')
    
     # Create summary table for sentiment counts
    st.subheader("Sentiment Summary")
   
    # Display the sentiment summary table
    st.table(summary_df)
# Footer
footer = """
    <div class="footer">
        <p style="text-align: left;">Copyright © 2024 MPSeDC. All rights reserved.</p>
        <p style="text-align: right;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)
