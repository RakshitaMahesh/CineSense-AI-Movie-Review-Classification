
import os
import pickle
import re
import html

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "artifacts/movie_review_lstm.keras"
WORD_INDEX_PATH = "artifacts/word_index.pkl"
VOCAB_SIZE = 10000
MAX_LENGTH = 200

st.set_page_config(
    page_title="CineSense AI | Movie Sentiment",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PROFESSIONAL DARK THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, #25204b 0%, transparent 36%),
        radial-gradient(ellipse at 95% 12%, #16334a 0%, transparent 32%),
        #090e1b;
    color: #f4f6ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.8px;
}

.brand span {
    color: #b9a1ff;
}

.nav-tag {
    background: #151e33;
    border: 1px solid #35415f;
    color: #d8dff0;
    border-radius: 30px;
    padding: 9px 13px;
    text-align: center;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.hero {
    text-align: center;
    padding: 55px 15px 32px;
}

.eyebrow {
    display: inline-block;
    padding: 9px 15px;
    border-radius: 30px;
    background: linear-gradient(120deg, #292044, #172d43);
    border: 1px solid #514777;
    color: #ded4ff;
    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(2.3rem, 5.2vw, 4.1rem);
    line-height: 1.13;
    letter-spacing: -2px;
    font-weight: 800;
    margin: 23px 0 18px;
}

.gradient-text {
    background: linear-gradient(90deg, #c4b5fd, #818cf8, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b1bdd4;
    font-size: 1rem;
    line-height: 1.85;
    max-width: 690px;
    margin: 0 auto;
}

.hero-note {
    color: #91a0bc;
    font-size: 0.82rem;
    margin-top: 18px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 18px 0 38px;
}

.stat-card {
    text-align: center;
    background: linear-gradient(145deg, #151f34, #10182a);
    border: 1px solid #2b3854;
    border-radius: 16px;
    padding: 20px 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.stat-card:hover {
    border-color: #8b7ad4;
    transform: translateY(-3px);
}

.stat-value {
    color: #e0d7ff;
    font-size: 1.45rem;
    font-weight: 800;
}

.stat-label {
    color: #a6b2ca;
    font-size: 0.79rem;
    margin-top: 7px;
}

.section-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 750;
    margin: 18px 0 6px;
}

.section-subtitle {
    color: #a6b2ca;
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 18px;
}

/* ---------- REVIEW TEXT BOX ---------- */
div[data-testid="stTextArea"] label {
    color: #dce3f5 !important;
    font-weight: 600 !important;
}

div[data-testid="stTextArea"] [data-baseweb="textarea"] {
    background: #111a2d !important;
    border: 1px solid #354361 !important;
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa !important;
}

div[data-testid="stTextArea"] textarea,
.stTextArea textarea {
    background: #111a2d !important;
    color: #f4f6ff !important;
    -webkit-text-fill-color: #f4f6ff !important;
    caret-color: #a78bfa !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 16px !important;
    font-size: 0.98rem !important;
    line-height: 1.75 !important;
    outline: none !important;
    box-shadow: none !important;
}

div[data-testid="stTextArea"] textarea::placeholder,
.stTextArea textarea::placeholder {
    color: #8290ad !important;
    -webkit-text-fill-color: #8290ad !important;
    opacity: 1 !important;
}

/* ---------- ALL BUTTONS ---------- */
.stButton > button {
    background: #151f34 !important;
    color: #f4f6ff !important;
    -webkit-text-fill-color: #f4f6ff !important;
    border: 1px solid #354361 !important;
    border-radius: 11px !important;
    min-height: 44px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #252443 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #a78bfa !important;
}

/* Primary Analyze button */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(100deg, #8b5cf6, #6366f1) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    min-height: 50px;
    font-size: 1rem;
    box-shadow: 0 5px 20px #6366f122;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(100deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    box-shadow: 0 7px 26px #6366f144;
}

/* ---------- CARDS ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111a2d;
    border: 1px solid #293752;
    border-radius: 16px;
}

.result-positive,
.result-negative {
    border-radius: 17px;
    padding: 25px;
    margin: 14px 0 20px;
    border: 1px solid;
}

.result-positive {
    background: linear-gradient(120deg, #102d2b, #122b30);
    border-color: #287565;
}

.result-negative {
    background: linear-gradient(120deg, #321d2b, #2b1c30);
    border-color: #85405b;
}

.result-kicker {
    color: #b7c4dc;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin-bottom: 9px;
}

.result-label {
    color: #ffffff;
    font-size: 1.65rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-positive .result-label {
    color: #6ee7b7;
}

.result-negative .result-label {
    color: #fda4af;
}

.result-detail {
    color: #c2cce0;
    font-size: 0.9rem;
    line-height: 1.7;
}

.metric-card {
    background: linear-gradient(145deg, #151f34, #101829);
    border: 1px solid #2b3955;
    padding: 20px;
    border-radius: 15px;
    min-height: 112px;
}

.metric-name {
    color: #a6b2ca;
    font-size: 0.82rem;
    margin-bottom: 12px;
}

.metric-value {
    color: #ffffff;
    font-size: 1.65rem;
    font-weight: 800;
}

/* ---------- HORIZONTAL WORKFLOW ---------- */
.pipeline {
    display: flex !important;
    flex-direction: row !important;
    justify-content: center !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 10px;
    background: #111a2d;
    border: 1px solid #293752;
    border-radius: 15px;
    padding: 20px 14px;
    margin: 15px 0;
}

.pipeline-step {
    display: inline-block !important;
    color: #e0d7ff !important;
    background: #202442 !important;
    border: 1px solid #45406f;
    padding: 10px 13px;
    border-radius: 9px;
    font-size: 0.83rem;
    font-weight: 600;
}

.pipeline-arrow {
    display: inline-block !important;
    color: #8290ad !important;
    font-size: 1.1rem;
}

/* ---------- PROJECT CARDS ---------- */
.info-card {
    background: linear-gradient(145deg, #141e32, #101829);
    border: 1px solid #2b3955;
    border-radius: 15px;
    padding: 21px;
    min-height: 158px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.info-card:hover {
    transform: translateY(-3px);
    border-color: #7467b1;
}

.info-card h4 {
    color: #f5f7ff;
    margin: 0 0 11px;
    font-size: 1rem;
}

.info-card p {
    color: #a6b2ca;
    font-size: 0.87rem;
    line-height: 1.75;
    margin: 0;
}

.footer {
    border-top: 1px solid #28334b;
    padding-top: 23px;
    margin-top: 44px;
    color: #8996b0;
    font-size: 0.82rem;
    line-height: 1.9;
    text-align: center;
}

.stAlert {
    border-radius: 12px;
}

@media (max-width: 700px) {
    .hero {
        padding: 38px 5px 25px;
    }

    .hero h1 {
        letter-spacing: -1px;
    }

    .stats-grid {
        gap: 8px;
    }

    .stat-card {
        padding: 15px 7px;
    }

    .stat-value {
        font-size: 1.15rem;
    }

    .stat-label {
        font-size: 0.7rem;
    }

    .pipeline {
        gap: 7px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<div class="brand">🎬 Cine<span>Sense</span> AI</div>',
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        '<div class="nav-tag">✦ DEEP LEARNING PROJECT</div>',
        unsafe_allow_html=True
    )

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ AI-POWERED SENTIMENT ANALYSIS</div>
    <h1>Every review tells a story.<br>
    <span class="gradient-text">Let AI read the emotion.</span></h1>
    <p>
        Discover the sentiment behind movie reviews using a deep learning
        model powered by Long Short-Term Memory networks.
    </p>
    <div class="hero-note">Understand the words. Discover the sentiment.</div>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">LSTM</div>
        <div class="stat-label">Deep Learning Architecture</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">IMDB</div>
        <div class="stat-label">Movie Review Dataset</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">NLP</div>
        <div class="stat-label">Natural Language Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_words():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(WORD_INDEX_PATH, "rb") as file:
        word_index = pickle.load(file)

    return model, word_index


# ---------------- PREPROCESS REVIEW ----------------
def encode_review(review, word_index):
    words = re.findall(r"[a-zA-Z0-9']+", review.lower())
    sequence = []

    for word in words:
        index = word_index.get(word)

        if index is not None and index + 3 < VOCAB_SIZE:
            sequence.append(index + 3)
        else:
            sequence.append(2)

    return pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


# ---------------- REVIEW ANALYZER ----------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""


def use_sample(text):
    st.session_state.review_text = text


st.markdown(
    '<div class="section-title">🎥 Analyze a movie review</div>'
    '<div class="section-subtitle">'
    'Write your own review or choose a sample to see the AI in action.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("**Try a sample review**")

sample1, sample2, sample3 = st.columns(3)

with sample1:
    st.button(
        "😊 Positive sample",
        use_container_width=True,
        on_click=use_sample,
        args=(
            "The movie was fantastic, the acting was brilliant, "
            "and I absolutely loved the story.",
        )
    )

with sample2:
    st.button(
        "😞 Negative sample",
        use_container_width=True,
        on_click=use_sample,
        args=(
            "The movie was boring and terrible. The acting was awful "
            "and I hated the ending.",
        )
    )

with sample3:
    st.button(
        "🎭 Mixed sample",
        use_container_width=True,
        on_click=use_sample,
        args=(
            "The story was interesting, but the acting was weak "
            "and the ending was disappointing.",
        )
    )

review = st.text_area(
    "YOUR MOVIE REVIEW",
    key="review_text",
    height=170,
    max_chars=5000,
    placeholder="Start typing your thoughts about the movie here..."
)

st.caption(
    "English reviews work best. Maximum length: 5,000 characters."
)

_, button_col, _ = st.columns([1, 2, 1])

with button_col:
    analyze = st.button(
        "✦  Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION RESULTS ----------------
if analyze:
    if not review.strip():
        st.warning("Please enter a movie review before analyzing.")

    elif not re.search(r"[a-zA-Z]", review):
        st.warning("Please enter a review containing English words.")

    elif not os.path.exists(MODEL_PATH) or not os.path.exists(WORD_INDEX_PATH):
        st.error("Model files not found. Please run python train_model.py first.")

    else:
        try:
            with st.spinner("🎬 CineSense AI is analyzing your review..."):
                model, word_index = load_model_and_words()
                encoded = encode_review(review, word_index)

                probability = float(
                    model.predict(encoded, verbose=0)[0][0]
                )

            positive_score = probability * 100
            negative_score = (1 - probability) * 100

            st.markdown("---")
            st.markdown(
                '<div class="section-title">✦ Your analysis</div>'
                '<div class="section-subtitle">'
                'Here is the sentiment predicted by your trained LSTM model.'
                '</div>',
                unsafe_allow_html=True
            )

            if probability >= 0.5:
                label = "Positive Review"
                emoji = "😊"
                css_class = "result-positive"
                explanation = (
                    "The model detected more positive sentiment in this review."
                )
            else:
                label = "Negative Review"
                emoji = "😞"
                css_class = "result-negative"
                explanation = (
                    "The model detected more negative sentiment in this review."
                )

            st.markdown(
                f'<div class="{css_class}">'
                f'<div class="result-kicker">SENTIMENT PREDICTION</div>'
                f'<div class="result-label">{emoji} {label}</div>'
                f'<div class="result-detail">{html.escape(explanation)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-name">💚 Positive sentiment score</div>'
                    f'<div class="metric-value">{positive_score:.2f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with metric2:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="metric-name">💗 Negative sentiment score</div>'
                    f'<div class="metric-value">{negative_score:.2f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")
            st.progress(float(np.clip(probability, 0.0, 1.0)))

            st.caption(
                "Scores represent the model's sigmoid output, not calibrated "
                "probabilities of correctness. Predictions may be less reliable "
                "for mixed, sarcastic, or unfamiliar reviews."
            )

        except Exception as error:
            st.error(f"Could not analyze the review: {error}")

# ---------------- HOW IT WORKS ----------------
st.markdown("---")

st.markdown(
    '<div class="section-title">⚙️ How CineSense AI works</div>'
    '<div class="section-subtitle">'
    'From written review to predicted sentiment in a simple pipeline.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="pipeline">
    <span class="pipeline-step">📝 Movie Review</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🔤 Text Processing</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🧠 LSTM Model</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">📊 Sentiment</span>
</div>
""", unsafe_allow_html=True)

# ---------------- PROJECT INFORMATION ----------------
st.markdown("---")

st.markdown(
    '<div class="section-title">🚀 Built with deep learning</div>'
    '<div class="section-subtitle">'
    'The technology behind the sentiment prediction.'
    '</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="info-card">
        <h4>🧠 LSTM Neural Network</h4>
        <p>
        Learns patterns in sequences of words to help classify
        movie reviews as positive or negative.
        </p>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <h4>📚 IMDB Dataset</h4>
        <p>
        Uses labeled movie reviews to train and evaluate a
        binary sentiment classification model.
        </p>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-card">
        <h4>⚡ Streamlit Interface</h4>
        <p>
        Provides an interactive web application for entering
        reviews and viewing sentiment predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong style="color:#d8dff0;">🎬 CineSense AI</strong><br>
    Movie Review Classification Using LSTM<br>
    Deep Learning Mini Project · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)
