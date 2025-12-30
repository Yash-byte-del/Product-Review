import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Amazon Review Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🛍️ Amazon Review Classifier")
st.markdown("### AI-Powered Product Review Analysis Using Deep Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Minimum confidence score to classify product as good"
    )
    
    min_reviews = st.number_input(
        "Minimum Reviews to Analyze",
        min_value=5,
        max_value=100,
        value=10,
        step=5
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses deep learning to analyze Amazon product reviews "
        "and determine product quality and review authenticity."
    )
    
    st.markdown("### 🔧 Technology Stack")
    st.markdown("""
    - **Streamlit** - Web framework
    - **Deep Learning** - Sentiment analysis
    - **NLP** - Text processing
    - **Plotly** - Data visualization
    """)

# Simulate deep learning sentiment analysis
def analyze_sentiment(text):
    """Simulate sentiment analysis using deep learning"""
    text_lower = text.lower()
    
    positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 
                     'best', 'awesome', 'fantastic', 'good', 'nice', 
                     'wonderful', 'outstanding', 'superb', 'quality']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 
                     'poor', 'disappointed', 'waste', 'broken', 'defective',
                     'useless', 'horrible', 'garbage', 'trash']
    
    score = 0
    for word in positive_words:
        if word in text_lower:
            score += 1
    for word in negative_words:
        if word in text_lower:
            score -= 1
    
    # Normalize to 0-1 range
    normalized_score = max(0, min(1, (score + 5) / 10))
    return normalized_score

def detect_fake_review(text, rating, verified):
    """Simulate fake review detection using ML"""
    # Simple heuristic simulation
    text_len = len(text.split())
    
    # Factors that might indicate fake review
    suspicious_score = 0
    
    # Very short reviews with 5 stars
    if text_len < 10 and rating == 5:
        suspicious_score += 0.3
    
    # Not verified purchase
    if not verified:
        suspicious_score += 0.2
    
    # Excessive positive words for low ratings or vice versa
    sentiment = analyze_sentiment(text)
    if (sentiment > 0.7 and rating <= 2) or (sentiment < 0.3 and rating >= 4):
        suspicious_score += 0.4
    
    # Random factor
    suspicious_score += np.random.uniform(0, 0.2)
    
    # Return True if likely authentic (low suspicious score)
    return suspicious_score < 0.5

def generate_mock_reviews(num_reviews=10):
    """Generate mock reviews for demonstration"""
    reviews_data = [
        {"rating": 5, "text": "Great product! Exactly what I needed. Amazing quality and fast shipping.", "verified": True},
        {"rating": 4, "text": "Good product but took a while to arrive. Quality is nice though.", "verified": True},
        {"rating": 5, "text": "Excellent! Best purchase this year. Highly recommend to everyone!", "verified": False},
        {"rating": 2, "text": "Disappointed with the quality. Not as described in the listing.", "verified": True},
        {"rating": 1, "text": "Terrible product. Broke after one week. Complete waste of money.", "verified": True},
        {"rating": 5, "text": "Perfect! Love it. Amazing value for money. Will buy again.", "verified": True},
        {"rating": 4, "text": "Pretty good overall. Minor issues but nothing major. Worth the price.", "verified": True},
        {"rating": 3, "text": "Average product. Does the job but nothing special. Expected more.", "verified": True},
        {"rating": 5, "text": "Outstanding quality! Exceeded my expectations. Five stars!", "verified": True},
        {"rating": 2, "text": "Poor build quality. Not durable at all. Would not recommend.", "verified": True},
        {"rating": 4, "text": "Good but not great. It works as advertised. Decent purchase.", "verified": True},
        {"rating": 5, "text": "Absolutely wonderful! Best thing I've bought online. Perfect!", "verified": False},
    ]
    
    return reviews_data[:num_reviews]

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    product_url = st.text_input(
        "🔗 Enter Amazon Product URL",
        placeholder="https://www.amazon.com/product/...",
        help="Paste the Amazon product URL here to analyze its reviews"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    analyze_button = st.button("🔍 Analyze Product", type="primary", use_container_width=True)

if analyze_button:
    if not product_url:
        st.error("⚠️ Please enter a product URL")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis steps
        status_text.text("🌐 Fetching product data...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        status_text.text("📥 Extracting reviews...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        status_text.text("🧠 Running deep learning model...")
        progress_bar.progress(60)
        time.sleep(0.8)
        
        status_text.text("📊 Analyzing sentiment and authenticity...")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # Generate mock reviews
        reviews = generate_mock_reviews(int(min_reviews))
        
        # Analyze each review
        analyzed_reviews = []
        for review in reviews:
            sentiment_score = analyze_sentiment(review['text'])
            is_authentic = detect_fake_review(review['text'], review['rating'], review['verified'])
            
            analyzed_reviews.append({
                'rating': review['rating'],
                'text': review['text'],
                'verified': review['verified'],
                'sentiment_score': sentiment_score,
                'authentic': is_authentic
            })
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        # Calculate statistics
        df = pd.DataFrame(analyzed_reviews)
        avg_rating = df['rating'].mean()
        avg_sentiment = df['sentiment_score'].mean()
        authentic_count = df['authentic'].sum()
        authentic_percent = (authentic_count / len(df)) * 100
        verified_count = df['verified'].sum()
        
        # Determine if product is good
        is_good = (avg_sentiment >= confidence_threshold and 
                   avg_rating >= 3.5 and 
                   authentic_percent >= 70)
        
        # Display overall verdict
        st.markdown("---")
        if is_good:
            st.success("### ✅ Good Product - Recommended")
            st.markdown(f"**Confidence:** {avg_sentiment*100:.1f}%")
        else:
            st.error("### ❌ Questionable Product - Not Recommended")
            st.markdown(f"**Confidence:** {avg_sentiment*100:.1f}%")
        
        st.markdown("---")
        
        # Metrics
        st.markdown("### 📈 Analysis Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
        with col3:
            st.metric("Authentic Reviews", f"{authentic_count}/{len(df)}")
        with col4:
            st.metric("Verified Purchases", f"{verified_count}/{len(df)}")
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Visual Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Rating distribution
            rating_counts = df['rating'].value_counts().sort_index()
            fig_rating = go.Figure(data=[
                go.Bar(x=rating_counts.index, y=rating_counts.values,
                      marker_color='steelblue')
            ])
            fig_rating.update_layout(
                title="Rating Distribution",
                xaxis_title="Rating",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with viz_col2:
            # Sentiment distribution
            fig_sentiment = go.Figure(data=[
                go.Histogram(x=df['sentiment_score'], nbinsx=10,
                           marker_color='teal')
            ])
            fig_sentiment.update_layout(
                title="Sentiment Score Distribution",
                xaxis_title="Sentiment Score",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Authenticity pie chart
        auth_data = pd.DataFrame({
            'Type': ['Authentic', 'Suspicious'],
            'Count': [authentic_count, len(df) - authentic_count]
        })
        fig_auth = px.pie(auth_data, values='Count', names='Type',
                         title='Review Authenticity',
                         color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig_auth, use_container_width=True)
        
        # Reviews table
        st.markdown("---")
        st.markdown("### 📝 Detailed Review Analysis")
        
        # Add sentiment labels
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: '😊 Positive' if x >= 0.7 else ('😐 Neutral' if x >= 0.4 else '😞 Negative')
        )
        
        # Display reviews
        for idx, row in df.iterrows():
            with st.expander(f"Review {idx+1} - {'⭐' * row['rating']} | {row['sentiment_label']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(row['text'])
                with col2:
                    if row['verified']:
                        st.markdown("🔵 **Verified**")
                    if row['authentic']:
                        st.markdown("✅ **Authentic**")
                    else:
                        st.markdown("⚠️ **Suspicious**")
                    st.markdown(f"**Sentiment:** {row['sentiment_score']:.2f}")
        
        # Note about demo
        st.markdown("---")
        st.info("""
        **📌 Demo Note:** This is a demonstration version with simulated data. 
        
        In a production environment, this would:
        - Scrape real Amazon reviews using Selenium/BeautifulSoup
        - Use pre-trained BERT/RoBERTa models for sentiment analysis
        - Employ advanced ML models for fake review detection
        - Include more sophisticated NLP techniques
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ using Streamlit | © 2024 Amazon Review Classifier"
    "</div>",
    unsafe_allow_html=True
)
