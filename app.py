"""
Streamlit UI for News Summarizer
Advanced AI Project - Media & Entertainment Domain
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
import sys

# Import modules
from model_loader import model_loader, ModelLoader
from summarizer import NewsSummarizer
from word2vec_search import SemanticSearch

# Page config
st.set_page_config(
    page_title="News Summarizer - AI Project",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .metric-card b {
        color: black !important;
    }
    .metric-card span {
        color: black !important;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=80)
    st.title("📰 News Summarizer")
    st.markdown("---")
    
    # Model loading
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI models..."):
            # Load model
            if model_loader.load_model():
                st.session_state.summarizer = NewsSummarizer(model_loader)
                
                # Load Word2Vec
                search_engine = SemanticSearch()
                if search_engine.load_embeddings():
                    st.session_state.search_engine = search_engine
                else:
                    st.warning("Word2Vec embeddings not found. Some features may be limited.")
                
                st.session_state.model_loaded = True
                st.success("✅ Models loaded successfully!")
    
    # Model info
    if st.session_state.model_loaded:
        info = model_loader.get_model_info()
        st.markdown("### Model Info")
        st.info(f"""
        - Status: {info['status']}
        - Device: {info['device']}
        - Type: {info['model_type']}
        """)
        
        # Navigation
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "Choose feature:",
            ["📝 Summarize", "🔍 Semantic Search", "📊 Analytics", "📚 History"]
        )
    else:
        page = "Loading..."
        st.warning("⚠️ Please wait for models to load")

# ============================================
# Main Content
# ============================================
st.markdown('<div class="main-header">📰 AI-Powered News Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Media & Entertainment Domain - Advanced AI Project</div>', unsafe_allow_html=True)

if not st.session_state.model_loaded:
    st.info("🔄 Loading AI models... This may take a few moments.")
    st.stop()

# ============================================
# PAGE: Summarize
# ============================================
if page == "📝 Summarize":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Enter Article Text")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Upload File", "Use Example"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            user_input = st.text_area(
                "Paste your news article here:",
                height=300,
                placeholder="Enter news article to summarize..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file:
                user_input = uploaded_file.read().decode('utf-8')
                st.text_area("File content:", user_input, height=200)
            else:
                user_input = ""
        
        else:  # Example
            examples = [
                "The UK economy showed strong growth in the third quarter as businesses reopened following the easing of lockdown restrictions. The Office for National Statistics reported that GDP grew by 15.5% between July and September, following a record contraction of 19.8% in the previous quarter. The services sector, which accounts for about 80% of economic output, grew by 14.2% in the third quarter. However, the economy remains 9.7% below its pre-pandemic level in February.",
                
                "Scientists have discovered a new species of dinosaur in Argentina that could be the largest ever found. The remains of the titanosaur, which lived about 100 million years ago, were unearthed in the Patagonia region. Initial estimates suggest the dinosaur weighed over 70 tons and measured 40 meters in length, surpassing the previous record holder, the Argentinosaurus."
            ]
            selected_example = st.selectbox("Select an example:", examples)
            user_input = selected_example
            st.text_area("Example content:", user_input, height=200)
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                max_length = st.slider("Max length", 50, 300, 150)
            with col_b:
                min_length = st.slider("Min length", 20, 100, 50)
            with col_c:
                num_beams = st.slider("Beams", 2, 8, 4)
            
            prompt_type = st.selectbox(
                "Prompt Strategy:",
                ["zero_shot", "few_shot", "chain_of_thought"],
                help="Different prompt engineering techniques"
            )
    
    with col2:
        st.markdown("### 📊 Statistics")
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            
            st.markdown(f"""
            <div class="metric-card">
                <b>📝 Word Count:</b> {word_count}<br>
                <b>📏 Character Count:</b> {char_count}<br>
                <b>📚 Estimated Reading Time:</b> {max(1, word_count//200)} min
            </div>
            """, unsafe_allow_html=True)
            
            # Summary button
            if st.button("🚀 Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    result = st.session_state.summarizer.summarize(
                        user_input,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        prompt_type=prompt_type
                    )
                    
                    if "error" not in result:
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "input": user_input[:100] + "...",
                            "summary": result["summary"],
                            "prompt": prompt_type,
                            "time": result["processing_time"]
                        })
                        
                        # Display results
                        st.markdown("### ✅ Summary")
                        st.markdown(f"""
                        <div class="success-box">
                            <b>📄 Generated Summary:</b><br>
                            {result['summary']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            st.metric("Input Length", f"{result['input_length']} words")
                        with col_met2:
                            st.metric("Output Length", f"{result['output_length']} words")
                        with col_met3:
                            st.metric("Processing Time", result['processing_time'])
                        
                        # Comparison
                        st.markdown("### 🔄 Compare All Prompt Strategies")
                        if st.button("Compare Strategies"):
                            with st.spinner("Comparing..."):
                                comparisons = st.session_state.summarizer.compare_prompts(user_input)
                                
                                for strategy, res in comparisons.items():
                                    if "error" not in res:
                                        col_a, col_b = st.columns([1, 2])
                                        with col_a:
                                            st.markdown(f"**{strategy}**")
                                            st.caption(f"Length: {res['output_length']} words")
                                        with col_b:
                                            st.write(res['summary'])
                                        st.divider()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            st.info("ℹ️ Enter an article to summarize")

# ============================================
# PAGE: Semantic Search
# ============================================
elif page == "🔍 Semantic Search":
    st.markdown("### 🔍 Semantic Article Search")
    st.caption("Find semantically similar news articles using Word2Vec")
    
    if st.session_state.search_engine:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter search query:",
                placeholder="e.g., political scandal government corruption"
            )
            
            top_n = st.slider("Number of results:", 3, 10, 5)
            
            if st.button("🔍 Search", type="primary"):
                if search_query:
                    with st.spinner("Searching..."):
                        results = st.session_state.search_engine.search(search_query, top_n=top_n)
                        
                        if results and "error" not in results[0]:
                            st.markdown(f"### ✅ Found {len(results)} articles")
                            
                            # Visualization
                            similarities = [r['similarity'] for r in results]
                            df_viz = pd.DataFrame({
                                'Rank': [r['rank'] for r in results],
                                'Similarity': similarities
                            })
                            fig = px.bar(df_viz, x='Rank', y='Similarity', 
                                        title='Similarity Scores')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Results
                            for r in results:
                                with st.expander(f"Rank {r['rank']} - Similarity: {r['similarity']:.4f}"):
                                    st.write(r['article'])
                        elif results:
                            st.warning("No similar articles found. Try a different query.")
                else:
                    st.warning("Please enter a search query")
        
        with col2:
            st.markdown("### 📊 Word2Vec Info")
            if st.session_state.search_engine.word2vec_model:
                vocab_size = len(st.session_state.search_engine.word2vec_model.wv)
                st.metric("Vocabulary Size", f"{vocab_size:,}")
                
                # Similar words
                st.markdown("### 🔤 Similar Words")
                word_input = st.text_input("Enter a word:", "economy")
                if word_input:
                    similar = st.session_state.search_engine.get_similar_words(word_input)
                    if similar:
                        for w, score in similar[:5]:
                            st.write(f"• {w} ({score:.3f})")
                    else:
                        st.caption("Word not found in vocabulary")
    else:
        st.warning("⚠️ Semantic search not available. Train Word2Vec embeddings first.")

# ============================================
# PAGE: Analytics
# ============================================
elif page == "📊 Analytics":
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Summaries", len(df_history))
        with col2:
            avg_time = df_history['time'].astype(str).str.replace('s', '').astype(float).mean()
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        with col3:
            prompt_counts = df_history['prompt'].value_counts()
            if not prompt_counts.empty:
                st.metric("Most Used Prompt", prompt_counts.index[0])
        
        # Visualization
        st.markdown("### 📈 Summary History")
        fig = px.line(df_history, x='timestamp', y=df_history.index,
                     title='Summary Timeline')
        st.plotly_chart(fig, use_container_width=True)
        
        # Prompt distribution
        st.markdown("### 📊 Prompt Strategy Distribution")
        prompt_dist = df_history['prompt'].value_counts()
        fig = px.pie(values=prompt_dist.values, names=prompt_dist.index,
                    title='Prompt Usage')
        st.plotly_chart(fig, use_container_width=True)
        
        # Word Cloud
        st.markdown("### ☁️ Summary Word Cloud")
        all_summaries = " ".join(df_history['summary'].tolist())
        if all_summaries:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_summaries)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        # History table
        with st.expander("📚 View Full History"):
            st.dataframe(df_history[['timestamp', 'input', 'summary', 'prompt']])
    else:
        st.info("ℹ️ No history yet. Generate some summaries first!")

# ============================================
# PAGE: History
# ============================================
elif page == "📚 History":
    st.markdown("### 📚 Summary History")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Summary #{len(st.session_state.history)-i} - {item['timestamp']}"):
                st.markdown(f"**Input:** {item['input']}")
                st.markdown(f"**Summary:** {item['summary']}")
                st.caption(f"Prompt: {item['prompt']} | Time: {item['time']}")
                
                if st.button(f"Delete #{len(st.session_state.history)-i}", key=f"del_{i}"):
                    st.session_state.history.pop(len(st.session_state.history)-i-1)
                    st.rerun()
        
        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("ℹ️ No history yet")

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <b>Advanced AI Project - Media & Entertainment Domain</b><br>
    Built with T5, Word2Vec, and Streamlit | © 2024
</div>
""", unsafe_allow_html=True)