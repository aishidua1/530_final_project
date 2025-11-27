# where i would put streamlit app
from pathlib import Path
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def ask_social_media_mental_health_bot(question: str) -> str:
    api_key = os.getenv("LITELLM_TOKEN")
    if not api_key:
        raise ValueError("LITELLM_TOKEN not found. Please set it first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://litellm.oit.duke.edu/v1"
    )

    user_prompt = (
        "The user has a question about mental health and social media use.\n"
        "Give a clear, concise, evidence-informed answer that a college student "
        "could understand.\n"
        "Be supportive but NOT therapeutic: do not diagnose or give medical advice.\n"
        "If the question sounds like the user might be in crisis, gently suggest "
        "contacting campus counseling or emergency services.\n\n"
        f"Question: {question}"
    )

    try:
        response = client.chat.completions.create(
            model="GPT 4.1 Mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supportive, non-clinical assistant who talks "
                        "about how social media impacts mental health using "
                        "balanced, research-informed information."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=400,
        )

        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]   # .. /final
DATA_PATH = ROOT / "data" / "student_social_media.csv"  # <-- change name if needed
IMG_DIR = ROOT / "imgs"

# ---------- DATA LOADING ----------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        # handles .xlsx, .xls
        return pd.read_excel(path)

df = load_data(DATA_PATH)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Student Social Media & Well-Being",
    layout="wide"
)

st.title("Student Social Media & Well-Being Dashboard")
st.write(
    """
    This dashboard explores relationships between **social media use**, **sleep**, and
    **mental health** in a student sample.  
    Use the sidebar to navigate through different views.
    """
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Pre-made Visualizations", "Build Your Own Plot", "Raw Data", "Chatbot"],
    )

    st.markdown("---")
    st.subheader("Filter data (optional)")
    # Generic example: filter by a column if it exists
    if "cluster" in df.columns:
        selected_clusters = st.multiselect(
            "Cluster", options=sorted(df["cluster"].dropna().unique())
        )
        if selected_clusters:
            df = df[df["cluster"].isin(selected_clusters)]

# ---------- PAGE 1: OVERVIEW ----------
if page == "Overview":
    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    # Number of students
    col1.metric("Number of students", len(df))

    numeric_cols = df.select_dtypes("number").columns

    if len(numeric_cols) > 0:
        # Simple overall mean of first numeric column, just as an example
        first_col = numeric_cols[0]
        col2.metric(f"Mean of {first_col}", f"{df[first_col].mean():.2f}")

        col3.metric("Number of numeric variables", len(numeric_cols))

    st.markdown("### Quick stats (numeric columns)")
    st.dataframe(df.describe().T)

    st.markdown("### Key Figures (Saved Images)")
    img_cols = st.columns(2)

    # Safely show images if they exist
    pre_made_imgs = [
        ("box_plot_social", "Distribution of social media use"),
        ("daily_smu_sleep", "Daily social media use vs. sleep"),
        ("mh_score_by_c", "Mental health score by group / cluster"),
        ("kmeans_cluster", "K-means clusters"),
        ("detailed_smu_s", "Detailed social media & sleep breakdown"),
    ]

    for i, (stem, caption) in enumerate(pre_made_imgs):
        img_path = next(IMG_DIR.glob(f"{stem}*"), None)
        if img_path is not None:
            with img_cols[i % 2]:
                st.image(str(img_path), use_container_width=True, caption=caption)

# ---------- PAGE 2: PRE-MADE VISUALIZATIONS ----------
elif page == "Pre-made Visualizations":
    st.subheader("Pre-made Visualizations (from your analysis)")

    # Show images one by one with explanation text placeholders
    def show_image(stem: str, title: str, explanation: str):
        img_path = next(IMG_DIR.glob(f"{stem}*"), None)
        if img_path is not None:
            st.markdown(f"#### {title}")
            st.image(str(img_path), use_container_width=True)
            st.write(explanation)
            st.markdown("---")

    show_image(
        "box_plot_social",
        "Social Media Use Distribution",
        "This box plot shows how daily social media use varies across students "
        "(median, spread, and potential outliers).",
    )

    show_image(
        "daily_smu_sleep",
        "Daily Social Media Use vs. Sleep",
        "This figure visualizes the relationship between hours of social media and sleep.",
    )

    show_image(
        "mh_score_by_c",
        "Mental Health by Group / Cluster",
        "This visualization compares mental health scores across different groups or clusters.",
    )

    show_image(
        "kmeans_cluster",
        "K-means Clusters",
        "The cluster plot groups students based on similar patterns in variables such as "
        "social media use, sleep, and mental health.",
    )

    show_image(
        "detailed_smu_s",
        "Detailed Social Media & Sleep Patterns",
        "A more granular breakdown of how social media use aligns with sleep behavior.",
    )

# ---------- PAGE 3: BUILD YOUR OWN PLOT ----------
elif page == "Build Your Own Plot":
    st.subheader("Explore the Data Interactively")

    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset.")
    else:
        tab1, tab2 = st.tabs(["Histogram / Boxplot", "Scatterplot"])

        with tab1:
            st.markdown("#### Histogram / Boxplot")
            col = st.selectbox("Choose a numeric variable", numeric_cols)

            plot_type = st.radio("Plot type", ["Histogram", "Boxplot"], horizontal=True)

            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                ax.hist(df[col].dropna(), bins=20)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram of {col}")
            else:
                ax.boxplot(df[col].dropna(), vert=True)
                ax.set_ylabel(col)
                ax.set_title(f"Boxplot of {col}")

            st.pyplot(fig)

        with tab2:
            st.markdown("#### Scatterplot")
            x_var = st.selectbox("X-axis", numeric_cols, index=0)
            y_var = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))

            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_var], df[y_var], alpha=0.6)
            ax2.set_xlabel(x_var)
            ax2.set_ylabel(y_var)
            ax2.set_title(f"{y_var} vs. {x_var}")
            st.pyplot(fig2)

# ---------- PAGE 4: RAW DATA ----------
elif page == "Raw Data":
    st.subheader("Raw Data")
    st.dataframe(df)

    st.download_button(
        "Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_social_media_sleep_filtered.csv",
        mime="text/csv",
    )


# chatbot
elif page == "Chatbot":

    st.header("Mental Health & Social Media Q&A Bot")

    # User input
    user_question = st.text_input("Ask a question about mental health and social media:")

    # When button is clicked
    if st.button("Ask"):
        if user_question.strip() == "":
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                answer = ask_social_media_mental_health_bot(user_question)
            
            st.subheader("Answer")
            st.write(answer)

    # st.subheader("Chat with the Dashboard Assistant")

    # tabs = st.tabs(["General Chat"])

    # # ------------- TAB 1: General Chat -------------
    # with tabs[0]:
    #     st.write("Ask any question about the dashboard, dataset, or insights.")

    #     if "chat_messages" not in st.session_state:
    #         st.session_state.chat_messages = [
    #             {"role": "assistant", "content": "Hi! How can I help you today?"}
    #         ]

    #     # Show chat history
    #     for msg in st.session_state.chat_messages:
    #         with st.chat_message(msg["role"]):
    #             st.markdown(msg["content"])

    #     # Chat input
    #     user_input = st.chat_input("Type your message here...")

    #     if user_input:
    #         # Log user message
    #         st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
    #         with st.chat_message("user"):
    #             st.markdown(user_input)

    #         # Call your LLM (ask_llm)
    #         with st.chat_message("assistant"):
    #             reply = summarize_reviews([user_input])  # temporarily reuse your LLM function
    #             st.markdown(reply)

    #         st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    # # ------------- TAB 2: Summarize Reviews -------------
    # with tabs[1]:
    #     st.write("Paste multiple reviews below, one per line, and get a summary.")

    #     review_text = st.text_area("Enter reviews here:", height=200)

    #     if st.button("Summarize Reviews"):
    #         if review_text.strip() == "":
    #             st.warning("Please enter at least one review.")
    #         else:
    #             reviews = [r.strip() for r in review_text.split("\n") if r.strip()]

    #             with st.spinner("Summarizing reviews..."):
    #                 summary = summarize_reviews(reviews)

    #             st.success("Summary:")
    #             st.write(summary)