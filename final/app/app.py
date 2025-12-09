from pathlib import Path
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from dotenv import load_dotenv

# ----------------- ENV SETUP -----------------
load_dotenv()


# ----------------- LLM HELPER -----------------
def ask_social_media_mental_health_bot(
    question: str,
    personalization_text: str | None = None,
) -> str:
    """
    Ask the mental health & social media bot a question.
    If personalization_text is provided, it will be included as context.
    """
    api_key = os.getenv("LITELLM_TOKEN")
    if not api_key:
        raise ValueError("LITELLM_TOKEN not found. Please set it first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://litellm.oit.duke.edu/v1",
    )

    # Base user prompt
    user_prompt = (
        "The user has a question about mental health and social media use.\n"
        "Give a clear, concise, evidence-informed answer that a college student "
        "could understand.\n"
        "Be supportive but NOT therapeutic: do not diagnose or give medical advice.\n"
        "If the question sounds like the user might be in crisis, gently suggest "
        "contacting campus counseling or emergency services.\n\n"
    )

    # Add personalization if available
    if personalization_text:
        user_prompt += (
            "Here is some additional context about the user's recent screen time. "
            "Use this to make your answer more concrete and personalized, but do NOT "
            "assume any diagnosis.\n\n"
            f"{personalization_text}\n\n"
        )

    user_prompt += f"Question: {question}"

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


# ----------------- PATHS & DATA -----------------
ROOT = Path(__file__).resolve().parents[1]   # .. /final
DATA_PATH = ROOT / "data" / "student_social_media.csv"
IMG_DIR = ROOT / "imgs"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)


df = load_data(DATA_PATH)


# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Student Social Media & Well-Being",
    layout="wide",
)

st.title("Student Social Media & Well-Being Dashboard")
st.write(
    """
    This dashboard explores relationships between **social media use**, **sleep**, and
    **mental health** in a student sample.

    On the **Home** page, you can:
    - Ask a **Q&A bot** about social media and mental health  
    - Enter your own **screen time patterns** to get more personalized insights  
    - Explore **pre-made visualizations** and summary statistics
    """
)


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Home", "Build Your Own Plot", "Raw Data"],
    )

    if "cluster" in df.columns:
        selected_clusters = st.multiselect(
            "Cluster", options=sorted(df["cluster"].dropna().unique())
        )
        if selected_clusters:
            df = df[df["cluster"].isin(selected_clusters)]


# ----------------- HOME: BOT + PRE-MADE VISUALS -----------------
if page == "Home":
    tab_bot, tab_viz = st.tabs(["Q&A Assistant", "Data & Visualizations"])

    # ---- TAB 1: BOT + MANUAL SCREEN TIME INPUT ----
    with tab_bot:
        st.header("Mental Health & Social Media Q&A Bot")

        st.markdown(
            """
            Ask a question about how social media, screen time, sleep, or online habits
            might relate to your mental health.  
            This assistant is **informational**, not clinical — it won't diagnose or
            replace professional care.
            """
        )

        st.markdown("#### See it in motion! Describe your typical screen time")

        st.write(
            """
            You can enter your **average daily screen time** below.  
            These numbers don’t need to be perfect — rough estimates are fine.
            The app will summarize them and use that context to personalize the answer.
            """
        )

        col_a, col_b = st.columns(2)
        with col_a:
            total_screen_minutes = st.number_input(
                "Average total screen time per day (minutes)",
                min_value=0,
                max_value=1440,
                value=0,
                step=10,
            )
        with col_b:
            social_media_minutes = st.number_input(
                "Average social media time per day (minutes)",
                min_value=0,
                max_value=1440,
                value=0,
                step=10,
            )

        days_pattern = st.number_input(
            "Roughly how many recent days does this pattern reflect?",
            min_value=1,
            max_value=365,
            value=7,
            step=1,
        )

        personalization_summary = None
        summary_parts = []

        if total_screen_minutes > 0:
            summary_parts.append(
                f"- Average total screen time: **{total_screen_minutes:.1f} minutes/day**"
            )
        if social_media_minutes > 0:
            summary_parts.append(
                f"- Average social media time: **{social_media_minutes:.1f} minutes/day**"
            )

        if summary_parts:
            summary_parts.append(
                f"- User reports this pattern over roughly **{days_pattern} days**"
            )

            st.markdown("###### Summary of your screen time")
            for line in summary_parts:
                st.write(line)

            personalization_summary = (
                "User-reported screen time pattern:\n" + "\n".join(summary_parts)
            )
        else:
            st.info(
                "If you enter non-zero values for screen time, I’ll use them to personalize "
                "the answer. You can also leave them at 0 and just ask a general question."
            )

        # Store in session so it’s available when the button is clicked
        st.session_state["screen_time_summary"] = personalization_summary

        st.markdown("#### Ask your question")
        user_question = st.text_area(
            "Type your question here:",
            placeholder="Example: How might using TikTok late at night affect my sleep and mood?",
            height=100,
        )

        if st.button("Ask", key="ask_bot_button"):
            if user_question.strip() == "":
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Thinking..."):
                    answer = ask_social_media_mental_health_bot(
                        user_question,
                        personalization_text=st.session_state.get("screen_time_summary"),
                    )

                st.subheader("Answer")
                st.write(answer)

    # ---- TAB 2: PRE-MADE VISUALIZATIONS + EXPLANATIONS ----
    with tab_viz:
        st.subheader("Dataset Summary & Pre-made Visualizations")

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of students", len(df))

        numeric_cols = df.select_dtypes("number").columns
        if len(numeric_cols) > 0:
            first_col = numeric_cols[0]
            col2.metric(f"Mean of {first_col}", f"{df[first_col].mean():.2f}")
            col3.metric("Number of numeric variables", len(numeric_cols))

        st.markdown("### Quick stats (numeric columns)")
        st.dataframe(df.describe().T)

        st.markdown("---")

        st.markdown(
        """
        ### Why The Following Visualizations Matter

        These charts help reveal important patterns in how students use social media,
        how much they sleep, and how these habits relate to well-being.  
        
        By examining trends in this dataset, you can:
        - See **real behavior patterns** that may mirror your own digital habits  
        - Identify **risk signals**, such as high social media use paired with short sleep  
        - Compare your personal screen-time inputs to broader trends  
        - Better understand how digital routines can influence **stress, mood, and overall wellness**  

        These visuals provide valuable context when interpreting your own habits and
        when asking questions in the Q&A Assistant. They are not diagnostic, but they
        help illustrate how certain technology patterns may support—or strain—mental health.
        """
    )



        st.markdown("---")
        st.markdown("### Visualizations and Insights")

        def show_image_with_insights(stem: str, title: str, insight_text: str):
            img_path = next(IMG_DIR.glob(f"{stem}*"), None)
            if img_path is not None:
                st.markdown(f"#### {title}")
                st.image(str(img_path), use_container_width=True)
                st.markdown(insight_text)
                st.markdown("---")

        show_image_with_insights(
            "box_plot_social",
            "Social Media Use Distribution",
            """
            **What this shows:**  
            This box plot summarizes how much time students spend on social media per day.  

            **How to read it:**  
            - The **median line** shows a typical daily usage.  
            - The **box height** reflects how spread out students' usage is.  
            - Any **points far above the box** are very heavy users.  

            **Why it matters:**  
            A wide spread or many outliers suggests that some students may be using social
            media at levels that could compete with sleep, studying, or offline activities.
            """,
        )

        show_image_with_insights(
            "daily_smu_sleep",
            "Daily Social Media Use vs. Sleep",
            """
            **What this shows:**  
            This figure compares hours of social media use with hours of sleep.  

            **Key ideas to look for:**  
            - If the trend slopes **downwards**, higher social media use might be linked
              to **less sleep**.  
            - Clusters of points in the “high social media / low sleep” area can hint at
              potentially risky patterns.  

            **Why it matters:**  
            Sleep is tightly connected to mood and concentration. When social media starts
            cutting into sleep, students may feel more tired, stressed, or emotionally
            reactive during the day.
            """,
        )

        show_image_with_insights(
            "mh_score_by_c",
            "Mental Health Score by Group / Cluster",
            """
            **What this shows:**  
            Students are grouped into clusters based on their behavior (e.g., social media,
            sleep, possibly other variables). Each bar reflects the average mental health
            score in that cluster.  

            **How to interpret it:**  
            - Clusters with **higher mental health scores** may represent healthier
              patterns of tech use and sleep.  
            - Clusters with **lower scores** might combine heavier late-night use,
              less sleep, or other stressors.  

            **Why it matters:**  
            Seeing these groups side by side helps illustrate that it's not just *how much*
            you use social media, but **how it fits into your day** that matters.
            """,
        )

        show_image_with_insights(
            "kmeans_cluster",
            "K-means Clusters of Students",
            """
            **What this shows:**  
            Each point represents a student, and colors show clusters of students with
            similar patterns across multiple variables (e.g., social media use, sleep,
            mental health scores).  

            **What to look for:**  
            - Are there clusters with **high social media + low sleep**?  
            - Are there clusters with **moderate use + good sleep** and better mental
              health scores?  

            **Why it matters:**  
            Clusters highlight that there’s more than one “type” of student tech behavior.
            Some patterns appear more protective while others may be more risky for mood
            and stress.
            """,
        )

        show_image_with_insights(
            "detailed_smu_s",
            "Detailed Social Media & Sleep Patterns",
            """
            **What this shows:**  
            A more granular breakdown of how different slices of social media behavior
            (e.g., evening use, total hours) align with sleep duration or quality.  

            **How to use it:**  
            - Focus on where **late-night social media** overlaps with shorter sleep.  
            - Look for any thresholds (e.g., past a certain number of hours, sleep
              tends to drop).  

            **Why it matters:**  
            Small changes, like moving heavy social media use earlier in the day or
            setting a “digital bedtime,” may help protect sleep and, indirectly,
            mental health.
            """,
        )


# ----------------- PAGE: BUILD YOUR OWN PLOT -----------------
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
            y_var = st.selectbox(
                "Y-axis",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
            )

            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_var], df[y_var], alpha=0.6)
            ax2.set_xlabel(x_var)
            ax2.set_ylabel(y_var)
            ax2.set_title(f"{y_var} vs. {x_var}")
            st.pyplot(fig2)


# ----------------- PAGE: RAW DATA -----------------
elif page == "Raw Data":
    st.subheader("Raw Data")
    st.dataframe(df)

    st.download_button(
        "Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_social_media_sleep_filtered.csv",
        mime="text/csv",
    )
