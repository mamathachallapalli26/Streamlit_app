# ============================================================
# MENTAL HEALTH IN TECH SURVEY
# EDA CAPSTONE PROJECT
# Author: Challapalli Mamatha
# Dataset: OSMI Mental Health in Tech Survey - 2014
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mental Health in Tech | EDA Capstone",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOR PALETTE
# ============================================================

PINE = "#2F6F62"
PINE_DARK = "#20504A"
PINE_LIGHT = "#B9D6CC"
AMBER = "#D98E3F"
RED = "#C0524A"
BLUE = "#4F7CAC"
PURPLE = "#8064A2"
GREEN = "#4C956C"
GRAY = "#C9CFCC"
DARK = "#1C2B2B"
MUTED = "#647270"
BACKGROUND = "#F5F7F4"
WHITE = "#FFFFFF"
GRID = "#DEE5E1"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BACKGROUND};
    }}

    [data-testid="stSidebar"] {{
        background-color: #EDF2EF;
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    h1, h2, h3, h4 {{
        color: {DARK};
    }}

    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {DARK};
        line-height: 1.15;
    }}

    .subtitle {{
        color: {MUTED};
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }}

    .section-title {{
        font-size: 1.8rem;
        font-weight: 750;
        color: {DARK};
        margin-top: 0.8rem;
    }}

    .section-text {{
        color: {MUTED};
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }}

    .kpi {{
        background: {WHITE};
        border: 1px solid {GRID};
        border-radius: 14px;
        padding: 1.25rem;
        min-height: 135px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }}

    .kpi-label {{
        color: {MUTED};
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}

    .kpi-value {{
        color: {PINE_DARK};
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }}

    .kpi-description {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }}

    .insight {{
        background: #E9F2EE;
        border-left: 4px solid {PINE};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: {DARK};
        line-height: 1.6;
    }}

    .warning {{
        background: #FFF4E5;
        border-left: 4px solid {AMBER};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: {DARK};
        line-height: 1.6;
    }}

    .recommendation {{
        background: {WHITE};
        border: 1px solid {GRID};
        border-left: 5px solid {PINE};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 7px rgba(0,0,0,0.03);
    }}

    .recommendation-title {{
        font-weight: 750;
        color: {DARK};
        font-size: 1.05rem;
    }}

    .recommendation-body {{
        color: {MUTED};
        line-height: 1.55;
        margin-top: 0.4rem;
    }}

    .about {{
        background: {WHITE};
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.65;
    }}

    .small {{
        color: {MUTED};
        font-size: 0.82rem;
    }}

    .footer {{
        text-align: center;
        color: {MUTED};
        border-top: 1px solid {GRID};
        padding-top: 1.5rem;
        margin-top: 3rem;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENDER NORMALIZATION
# ============================================================

MALE_SET = {
    "male",
    "m",
    "man",
    "cis male",
    "male (cis)",
    "cis man",
    "mal",
    "maile",
    "make",
    "malr",
    "msle",
    "guy (-ish) ^_^",
    "male-ish",
    "something kinda male?",
}

FEMALE_SET = {
    "female",
    "f",
    "woman",
    "cis female",
    "cis-female/femme",
    "femake",
    "female (cis)",
    "femail",
    "female ",
}


def normalize_gender(value):

    value = str(value).strip().lower()

    if value in MALE_SET:
        return "Male"

    if value in FEMALE_SET:
        return "Female"

    return "Other / non-binary"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    path = "survey.csv"

    if not os.path.exists(path):
        return None

    data = pd.read_csv(path)

    # Standardize column names
    data.columns = [
        str(col).strip()
        for col in data.columns
    ]

    # Gender cleaning
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].apply(
            normalize_gender
        )

    # Age cleaning
    if "Age" in data.columns:

        data["Age"] = pd.to_numeric(
            data["Age"],
            errors="coerce"
        )

        data["Age"] = data["Age"].where(
            data["Age"].between(15, 80)
        )

    # Fill selected missing values
    if "self_employed" in data.columns:

        data["self_employed"] = (
            data["self_employed"]
            .fillna("No")
        )

    if "work_interfere" in data.columns:

        data["work_interfere"] = (
            data["work_interfere"]
            .fillna("Not applicable")
        )

    # Remove columns that are not useful
    # for the main EDA
    for col in [
        "Timestamp",
        "comments",
        "state",
    ]:

        if col in data.columns:
            data = data.drop(
                columns=col
            )

    # Remove invalid ages
    if "Age" in data.columns:

        data = data.dropna(
            subset=["Age"]
        )

    return data


# ============================================================
# DATA INITIALIZATION
# ============================================================

df = load_data()

if df is None:

    st.error(
        "survey.csv was not found."
    )

    st.info(
        "Place survey.csv in the same folder as app.py."
    )

    st.stop()


N = len(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pct(condition):

    if len(df) == 0:
        return 0

    return round(
        condition.mean() * 100,
        1
    )


# ------------------------------------------------------------
# GLOBAL SUMMARY RATES
# (computed once so every page can use them, not just Overview)
# ------------------------------------------------------------

treatment_rate = pct(
    df["treatment"] == "Yes"
)

family_rate = pct(
    df["family_history"] == "Yes"
)

work_rate = pct(
    df["work_interfere"].isin(
        ["Sometimes", "Often"]
    )
)

benefit_rate = pct(
    df["benefits"] == "Yes"
)


def clean_value_counts(column):

    if column not in df.columns:
        return pd.DataFrame()

    result = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    result.columns = [
        "Category",
        "Count"
    ]

    return result


def base_chart(fig, height=430):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(
            family="Arial",
            color=DARK,
            size=13
        ),
        margin=dict(
            l=40,
            r=30,
            t=65,
            b=45
        ),
        title_font=dict(
            size=18,
            color=DARK
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=GRID
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    return fig


def show_chart(
    fig,
    why,
    insight,
    impact,
    height=430
):

    fig = base_chart(
        fig,
        height
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )

    with st.expander(
        "📊 Chart interpretation"
    ):

        st.markdown(
            f"**Why this chart?** {why}"
        )

        st.markdown(
            f"**Insight:** {insight}"
        )

        st.markdown(
            f"**Business impact:** {impact}"
        )


def treatment_rate_by(column):

    if (
        column not in df.columns
        or "treatment" not in df.columns
    ):
        return pd.DataFrame()

    result = (
        df.groupby(column)["treatment"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index()
    )

    result.columns = [
        "Category",
        "Treatment Rate"
    ]

    return result.sort_values(
        "Treatment Rate",
        ascending=False
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Mental Health in Tech"
    )

    st.caption(
        "OSMI Survey · EDA Capstone"
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Know Your Data",
            "👥 Univariate Analysis",
            "🔗 Bivariate Analysis",
            "🧩 Multivariate Analysis",
            "🎯 Business Insights",
            "📌 Recommendations & About",
        ]
    )

    st.markdown("---")

    st.markdown(
        "**Dataset**"
    )

    st.caption(
        f"{N:,} cleaned respondents"
    )

    st.caption(
        "OSMI Mental Health in Tech Survey"
    )

    st.markdown("---")

    st.caption(
        "Built by"
    )

    st.markdown(
        "**Challapalli Mamatha**"
    )


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    "🧠 Mental Health in Tech Survey"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Exploratory Data Analysis · Visualization · Business Insights"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">'
        "Project Overview"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "This dashboard explores the OSMI Mental Health in Tech Survey "
        "to understand mental-health treatment, workplace interference, "
        "employee support and attitudes toward mental health."
        "</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            "RESPONDENTS",
            f"{N:,}",
            "After data cleaning"
        ),
        (
            "SOUGHT TREATMENT",
            f"{treatment_rate:.0f}%",
            "Reported treatment"
        ),
        (
            "FAMILY HISTORY",
            f"{family_rate:.0f}%",
            "Reported family history"
        ),
        (
            "WORK INTERFERENCE",
            f"{work_rate:.0f}%",
            "Sometimes or often"
        ),
    ]

    for col, card in zip(
        [c1, c2, c3, c4],
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div class="kpi">

                    <div class="kpi-label">
                        {card[0]}
                    </div>

                    <div class="kpi-value">
                        {card[1]}
                    </div>

                    <div class="kpi-description">
                        {card[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")

    # Overview chart 1
    treatment_counts = (
        df["treatment"]
        .value_counts()
        .reset_index()
    )

    treatment_counts.columns = [
        "Treatment",
        "Count"
    ]

    fig = px.pie(
        treatment_counts,
        names="Treatment",
        values="Count",
        hole=0.55,
        color="Treatment",
        color_discrete_map={
            "Yes": PINE,
            "No": GRAY
        },
        title="Treatment Distribution"
    )

    show_chart(
        fig,
        "A donut chart gives a simple overview of the target variable.",
        "The survey contains both respondents who reported treatment "
        "and respondents who reported no treatment.",
        "Understanding the target distribution is important before "
        "building any predictive or classification model."
    )

    # Overview chart 2
    if "Country" in df.columns:

        countries = (
            df["Country"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        fig = px.bar(
            x=countries.values,
            y=countries.index,
            orientation="h",
            title="Top 10 Respondent Countries",
            labels={
                "x": "Respondents",
                "y": "Country"
            },
            color_discrete_sequence=[
                PINE
            ]
        )

        show_chart(
            fig,
            "A horizontal bar chart makes country-level comparison easy.",
            "The survey has a strong concentration of respondents "
            "from a limited number of countries.",
            "Geographic concentration is important when deciding how "
            "far the findings can be generalized."
        )


# ============================================================
# PAGE 2 — KNOW YOUR DATA
# ============================================================

elif page == "📊 Know Your Data":

    st.markdown(
        '<div class="section-title">'
        "1. Know Your Data"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "### Dataset First View"
    )

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.markdown(
        "### Dataset Rows & Columns"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    st.markdown(
        "### Dataset Information"
    )

    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique())
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        info,
        width="stretch",
        height=450
    )

    st.markdown(
        "### Missing Values"
    )

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        fig = px.bar(
            x=missing.values,
            y=missing.index,
            orientation="h",
            title="Missing Values by Column",
            labels={
                "x": "Missing Values",
                "y": "Column"
            },
            color_discrete_sequence=[
                AMBER
            ]
        )

        show_chart(
            fig,
            "A horizontal bar chart quickly identifies variables "
            "with the largest amount of missing data.",
            "Some survey variables contain missing responses because "
            "not every respondent answered every question.",
            "Missing-value analysis helps prevent misleading conclusions "
            "and supports appropriate preprocessing."
        )

    else:

        st.success(
            "No missing values detected in the cleaned dataset."
        )

    st.markdown(
        "### Descriptive Statistics"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        width="stretch"
    )

    st.markdown(
        """
        <div class="insight">
        <strong>What was learned:</strong>
        The dataset contains demographic, workplace and mental-health
        variables. Data preparation is necessary because the original
        survey contains inconsistent categorical responses, missing
        values and free-text entries.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 3 — UNIVARIATE ANALYSIS
# ============================================================

elif page == "👥 Univariate Analysis":

    st.markdown(
        '<div class="section-title">'
        "2. Univariate Analysis"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "Univariate analysis examines individual variables to understand "
        "their distribution before studying relationships between variables."
        "</div>",
        unsafe_allow_html=True
    )

    # Chart 1 — Gender
    if "Gender" in df.columns:

        counts = clean_value_counts(
            "Gender"
        )

        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            title="Chart 1 — Gender Distribution",
            color_discrete_sequence=[
                PINE
            ]
        )

        show_chart(
            fig,
            "A bar chart is appropriate for comparing categorical groups.",
            "The dataset contains a larger representation of some gender "
            "groups than others.",
            "Understanding sample composition helps avoid overgeneralizing "
            "findings to underrepresented groups."
        )

    # Chart 2 — Age
    if "Age" in df.columns:

        fig = px.histogram(
            df,
            x="Age",
            nbins=25,
            title="Chart 2 — Age Distribution",
            color_discrete_sequence=[
                PINE
            ]
        )

        show_chart(
            fig,
            "A histogram is useful for understanding the distribution "
            "of a numerical variable.",
            f"The median respondent age is approximately "
            f"{df['Age'].median():.0f} years.",
            "Age distribution provides context for interpreting "
            "demographic patterns in the survey."
        )

    # Chart 3 — Treatment
    counts = clean_value_counts(
        "treatment"
    )

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        hole=0.45,
        title="Chart 3 — Treatment Status",
        color="Category",
        color_discrete_map={
            "Yes": PINE,
            "No": GRAY
        }
    )

    show_chart(
        fig,
        "A donut chart clearly communicates the share of each target class.",
        f"Approximately {treatment_rate:.0f}% of respondents reported "
        "seeking treatment.",
        "Target distribution is important for later classification "
        "and model evaluation."
    )

    # Chart 4 — Family history
    counts = clean_value_counts(
        "family_history"
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title="Chart 4 — Family History of Mental Illness",
        color_discrete_sequence=[
            BLUE
        ]
    )

    show_chart(
        fig,
        "A categorical bar chart compares the two response groups.",
        "A considerable portion of respondents reported a family history "
        "of mental illness.",
        "Family history can help organizations understand the diversity "
        "of experiences represented in the survey."
    )

    # Chart 5 — Work interference
    counts = clean_value_counts(
        "work_interfere"
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title="Chart 5 — Mental Health Work Interference",
        color_discrete_sequence=[
            AMBER
        ]
    )

    show_chart(
        fig,
        "A bar chart allows comparison across the ordered frequency "
        "categories.",
        "Respondents report different levels of work interference, "
        "ranging from never to often.",
        "Understanding work interference can help organizations "
        "identify areas where workplace support may be useful."
    )

    # Chart 6 — Benefits
    counts = clean_value_counts(
        "benefits"
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title="Chart 6 — Employer Mental Health Benefits",
        color_discrete_sequence=[
            GREEN
        ]
    )

    show_chart(
        fig,
        "A bar chart is effective for comparing employee responses "
        "about employer benefits.",
        "Respondents differ in whether they report having mental-health "
        "benefits through their employer.",
        "Benefit availability and awareness are relevant considerations "
        "for employee-support programs."
    )

    # Chart 7 — Care options
    counts = clean_value_counts(
        "care_options"
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title="Chart 7 — Awareness of Care Options",
        color_discrete_sequence=[
            PURPLE
        ]
    )

    show_chart(
        fig,
        "Categorical counts reveal how many respondents know about "
        "available care options.",
        "Not all respondents report being aware of workplace care options.",
        "Improving communication about available resources may reduce "
        "information gaps."
    )

    # Chart 8 — Company size
    counts = clean_value_counts(
        "no_employees"
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title="Chart 8 — Company Size Distribution",
        color_discrete_sequence=[
            PINE_DARK
        ]
    )

    show_chart(
        fig,
        "A bar chart allows comparison between company-size categories.",
        "The survey includes respondents from organizations of different "
        "sizes.",
        "Company size provides useful context when comparing workplace "
        "policies and support."
    )


# ============================================================
# PAGE 4 — BIVARIATE ANALYSIS
# ============================================================

elif page == "🔗 Bivariate Analysis":

    st.markdown(
        '<div class="section-title">'
        "3. Bivariate Analysis"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "Bivariate analysis examines relationships between two variables."
        "</div>",
        unsafe_allow_html=True
    )

    # Chart 9
    if "family_history" in df.columns:

        result = treatment_rate_by(
            "family_history"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 9 — Treatment Rate by Family History",
            color="Treatment Rate",
            color_continuous_scale=[
                PINE_LIGHT,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "Treatment rate comparison is more informative than raw counts "
            "when category sizes differ.",
            "Treatment rates differ between respondents with and without "
            "a reported family history.",
            "This information can help analysts identify variables "
            "associated with treatment-seeking."
        )

    # Chart 10
    if "work_interfere" in df.columns:

        result = treatment_rate_by(
            "work_interfere"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 10 — Treatment Rate by Work Interference",
            color="Treatment Rate",
            color_continuous_scale=[
                AMBER,
                RED
            ]
        )

        show_chart(
            fig,
            "Comparing treatment rates across work-interference categories "
            "shows how the two variables vary together.",
            "Treatment rates are not identical across levels of work "
            "interference.",
            "The pattern can help organizations understand the relationship "
            "between workplace functioning and treatment-seeking."
        )

    # Chart 11
    if "benefits" in df.columns:

        result = treatment_rate_by(
            "benefits"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 11 — Treatment Rate by Employer Benefits",
            color="Treatment Rate",
            color_continuous_scale=[
                PINE_LIGHT,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "Treatment rates can be compared across employer benefit groups.",
            "Respondents reporting different benefit availability have "
            "different treatment rates.",
            "The result can inform further investigation into whether "
            "benefit awareness and accessibility are related to help-seeking."
        )

    # Chart 12
    if "care_options" in df.columns:

        result = treatment_rate_by(
            "care_options"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 12 — Treatment Rate by Care Options",
            color="Treatment Rate",
            color_continuous_scale=[
                BLUE,
                PURPLE
            ]
        )

        show_chart(
            fig,
            "This chart compares treatment rates for different levels "
            "of care-option awareness.",
            "Treatment rates vary according to respondents' reported "
            "awareness of care options.",
            "Organizations can use such analysis to investigate whether "
            "communication about support resources is reaching employees."
        )

    # Chart 13
    if "leave" in df.columns:

        result = treatment_rate_by(
            "leave"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 13 — Treatment Rate by Ease of Mental Health Leave",
            color="Treatment Rate",
            color_continuous_scale=[
                GREEN,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "Treatment rates across leave categories can highlight "
            "differences in workplace support experiences.",
            "Respondents report varying levels of difficulty when taking "
            "mental-health-related leave.",
            "Clear leave policies may be an important area for organizations "
            "to examine."
        )

    # Chart 14
    if "anonymity" in df.columns:

        result = treatment_rate_by(
            "anonymity"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 14 — Treatment Rate by Perceived Anonymity",
            color="Treatment Rate",
            color_continuous_scale=[
                BLUE,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "A grouped treatment-rate comparison shows whether reported "
            "anonymity differs alongside treatment.",
            "Treatment rates vary across responses about workplace anonymity.",
            "Transparent communication about privacy may be relevant "
            "when designing support programs."
        )

    # Chart 15
    if "remote_work" in df.columns:

        result = treatment_rate_by(
            "remote_work"
        )

        fig = px.bar(
            result,
            x="Category",
            y="Treatment Rate",
            title="Chart 15 — Treatment Rate by Remote Work",
            color="Treatment Rate",
            color_continuous_scale=[
                PINE_LIGHT,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "A simple comparison allows treatment rates to be compared "
            "between remote-work groups.",
            "Treatment rates differ across remote-work responses.",
            "Remote-work status may provide useful context when studying "
            "employee support and workplace experience."
        )


# ============================================================
# PAGE 5 — MULTIVARIATE ANALYSIS
# ============================================================

elif page == "🧩 Multivariate Analysis":

    st.markdown(
        '<div class="section-title">'
        "4. Multivariate Analysis"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "Multivariate analysis considers multiple variables together "
        "to uncover broader patterns in the survey."
        "</div>",
        unsafe_allow_html=True
    )

    # Chart 16 — Treatment by family history and gender
    if all(
        col in df.columns
        for col in [
            "family_history",
            "Gender",
            "treatment"
        ]
    ):

        grouped = (
            df.groupby(
                [
                    "family_history",
                    "Gender"
                ]
            )["treatment"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index()
        )

        grouped.columns = [
            "Family History",
            "Gender",
            "Treatment Rate"
        ]

        fig = px.bar(
            grouped,
            x="Family History",
            y="Treatment Rate",
            color="Gender",
            barmode="group",
            title="Chart 16 — Treatment by Family History and Gender",
            color_discrete_sequence=[
                PINE,
                AMBER,
                BLUE
            ]
        )

        show_chart(
            fig,
            "A grouped bar chart allows two categorical variables "
            "to be compared against treatment rate.",
            "Treatment patterns differ across combinations of family "
            "history and gender categories.",
            "Multivariate comparisons provide more context than examining "
            "each variable independently."
        )

    # Chart 17 — Benefits + care options
    if all(
        col in df.columns
        for col in [
            "benefits",
            "care_options",
            "treatment"
        ]
    ):

        pivot = (
            df.groupby(
                [
                    "benefits",
                    "care_options"
                ]
            )["treatment"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index()
        )

        pivot.columns = [
            "Benefits",
            "Care Options",
            "Treatment Rate"
        ]

        fig = px.scatter(
            pivot,
            x="Benefits",
            y="Care Options",
            size="Treatment Rate",
            color="Treatment Rate",
            hover_data=[
                "Treatment Rate"
            ],
            title="Chart 17 — Benefits and Care Options",
            color_continuous_scale=[
                PINE_LIGHT,
                PINE_DARK
            ]
        )

        show_chart(
            fig,
            "A multivariate visualization shows how two workplace "
            "support variables relate to treatment rates.",
            "Treatment rates vary across combinations of reported benefits "
            "and care-option awareness.",
            "This can help identify combinations that deserve further "
            "organizational investigation."
        )

    # Chart 18 — Company size + treatment
    if all(
        col in df.columns
        for col in [
            "no_employees",
            "treatment",
            "family_history"
        ]
    ):

        grouped = (
            df.groupby(
                [
                    "no_employees",
                    "family_history"
                ]
            )["treatment"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index()
        )

        grouped.columns = [
            "Company Size",
            "Family History",
            "Treatment Rate"
        ]

        fig = px.bar(
            grouped,
            x="Company Size",
            y="Treatment Rate",
            color="Family History",
            barmode="group",
            title="Chart 18 — Company Size, Family History and Treatment",
            color_discrete_sequence=[
                PINE,
                AMBER
            ]
        )

        show_chart(
            fig,
            "A grouped chart reveals how company size and family history "
            "interact with treatment rates.",
            "Treatment patterns can vary across both organization size "
            "and family-history groups.",
            "This can help prevent one-size-fits-all interpretations "
            "of workplace mental-health data."
        )

    # Chart 19 — Heatmap
    st.markdown(
        "### Chart 19 — Categorical Treatment Heatmap"
    )

    heat_columns = [
        col
        for col in [
            "family_history",
            "benefits",
            "care_options",
            "anonymity",
            "leave",
            "remote_work"
        ]
        if col in df.columns
    ]

    if len(heat_columns) > 1:

        heat_data = []

        for col in heat_columns:

            result = treatment_rate_by(
                col
            )

            for _, row in result.iterrows():

                heat_data.append(
                    {
                        "Variable": col,
                        "Category": row["Category"],
                        "Treatment Rate":
                            row["Treatment Rate"]
                    }
                )

        heat_df = pd.DataFrame(
            heat_data
        )

        matrix = heat_df.pivot_table(
            index="Variable",
            columns="Category",
            values="Treatment Rate"
        )

        fig = px.imshow(
            matrix,
            color_continuous_scale=[
                PINE_LIGHT,
                PINE_DARK
            ],
            aspect="auto",
            title="Treatment Rate Heatmap"
        )

        show_chart(
            fig,
            "A heatmap is useful for comparing treatment rates across "
            "many categorical variables simultaneously.",
            "The visualization highlights categories with relatively "
            "higher or lower observed treatment rates.",
            "It provides a compact way to identify areas that may deserve "
            "deeper analysis."
        )

    # Chart 20 — Correlation
    st.markdown(
        "### Chart 20 — Numerical Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[
                BLUE,
                WHITE,
                RED
            ],
            zmin=-1,
            zmax=1,
            title="Numerical Feature Correlation"
        )

        show_chart(
            fig,
            "A correlation heatmap summarizes relationships between "
            "multiple numerical variables.",
            "Correlation values show the direction and strength of "
            "linear association between numerical features.",
            "Correlation analysis can guide feature selection and "
            "identify variables worth investigating further."
        )

    # Chart 21 — Pair plot style
    st.markdown(
        "### Chart 21 — Numerical Relationship Explorer"
    )

    numeric_columns = list(
        numeric_df.columns
    )

    if len(numeric_columns) >= 2:

        x_num = st.selectbox(
            "X variable",
            numeric_columns,
            key="multi_x"
        )

        y_num = st.selectbox(
            "Y variable",
            numeric_columns,
            index=min(
                1,
                len(numeric_columns)-1
            ),
            key="multi_y"
        )

        fig = px.scatter(
            df,
            x=x_num,
            y=y_num,
            color=(
                "treatment"
                if "treatment" in df.columns
                else None
            ),
            title=f"{x_num} vs {y_num}",
            color_discrete_sequence=[
                PINE,
                AMBER
            ]
        )

        show_chart(
            fig,
            "A scatter plot is suitable for examining relationships "
            "between two numerical variables.",
            "The chart shows how observations are distributed across "
            "the selected numerical variables.",
            "Relationship exploration can reveal patterns and potential "
            "features for future statistical or machine-learning analysis."
        )


# ============================================================
# PAGE 6 — BUSINESS INSIGHTS
# ============================================================

elif page == "🎯 Business Insights":

    st.markdown(
        '<div class="section-title">'
        "5. Solution to Business Objective"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "The following observations translate the exploratory analysis "
        "into practical business considerations."
        "</div>",
        unsafe_allow_html=True
    )

    # Dynamic insights
    fam_yes = np.nan
    fam_no = np.nan

    if "family_history" in df.columns:

        fam_yes = (
            df[
                df["family_history"] == "Yes"
            ]["treatment"]
            .eq("Yes")
            .mean() * 100
        )

        fam_no = (
            df[
                df["family_history"] == "No"
            ]["treatment"]
            .eq("Yes")
            .mean() * 100
        )

    st.markdown(
        f"""
        <div class="insight">

        <strong>Finding 1 — Family history:</strong><br>

        Among respondents with a reported family history, approximately
        {fam_yes:.0f}% reported treatment, compared with approximately
        {fam_no:.0f}% among respondents without a reported family history.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight">

        <strong>Finding 2 — Workplace interference:</strong><br>

        Approximately {work_rate:.0f}% of respondents reported that
        mental health sometimes or often interfered with their work.

        </div>
        """,
        unsafe_allow_html=True
    )

    care_unknown = 0

    if "care_options" in df.columns:

        care_unknown = pct(
            df["care_options"].isin(
                [
                    "No",
                    "Not sure"
                ]
            )
        )

    st.markdown(
        f"""
        <div class="insight">

        <strong>Finding 3 — Awareness of care:</strong><br>

        Approximately {care_unknown:.0f}% of respondents selected
        "No" or "Not sure" regarding available care options.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Suggested Business Actions"
    )

    actions = [
        (
            "Increase awareness",
            "Make information about mental-health benefits and support "
            "resources easy to find and understand."
        ),
        (
            "Improve transparency",
            "Clearly explain confidentiality, anonymity and the process "
            "for accessing support."
        ),
        (
            "Review leave policies",
            "Ensure employees understand available leave and workplace "
            "support procedures."
        ),
        (
            "Train managers",
            "Provide appropriate workplace training so managers know "
            "how to direct employees toward support resources."
        ),
        (
            "Measure continuously",
            "Repeat anonymous employee surveys to evaluate changes "
            "in awareness and workplace experience."
        ),
    ]

    for i, (title, body) in enumerate(
        actions,
        1
    ):

        st.markdown(
            f"""
            <div class="recommendation">

                <div class="recommendation-title">
                    {i}. {title}
                </div>

                <div class="recommendation-body">
                    {body}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="warning">

        <strong>Important:</strong>

        These recommendations are based on associations observed in a
        self-selected survey. They should be treated as analytical
        considerations rather than evidence that a particular workplace
        policy directly causes a particular outcome.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 7 — RECOMMENDATIONS & ABOUT
# ============================================================

elif page == "📌 Recommendations & About":

    st.markdown(
        '<div class="section-title">'
        "📌 Recommendations & About"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        "Project information, business objective, recommendations and conclusion."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROJECT NAME
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Project Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="about">

            <strong>Project Name</strong>

            <br><br>

            Mental Health in Tech Survey

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="about">

            <strong>Project Type</strong>

            <br><br>

            Exploratory Data Analysis

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="about">

            <strong>Contribution</strong>

            <br><br>

            Individual Project

            </div>
            """,
            unsafe_allow_html=True
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="about">

            <strong>Author</strong>

            <br><br>

            Challapalli Mamatha

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="about">

            <strong>Dataset</strong>

            <br><br>

            OSMI Mental Health in Tech Survey

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="about">

            <strong>Technology</strong>

            <br><br>

            Python · Pandas · NumPy · Plotly · Streamlit

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### 📝 Project Summary"
    )

    st.markdown(
        """
        <div class="about">

        The <strong>Mental Health in Tech Survey</strong> project focuses
        on understanding mental-health experiences within the technology
        workplace using the OSMI Mental Health in Tech Survey dataset.

        <br><br>

        The project follows a structured Exploratory Data Analysis
        approach beginning with data loading, inspection, cleaning and
        preparation. Variables are examined using univariate,
        bivariate and multivariate analysis to identify meaningful
        patterns in the survey responses.

        <br><br>

        The analysis focuses on treatment-seeking behavior and its
        relationship with variables including family history,
        workplace interference, employer benefits, care options,
        anonymity, mental-health leave, remote work and company size.

        <br><br>

        More than twenty visual analyses are incorporated into the
        dashboard. These include categorical distributions,
        histograms, treatment-rate comparisons, grouped charts,
        heatmaps and numerical relationship analysis.

        <br><br>

        The purpose is to convert the raw survey responses into
        understandable analytical findings that can support discussion
        around workplace mental-health awareness, accessibility and
        employee support.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROBLEM STATEMENT
    # --------------------------------------------------------

    st.markdown(
        "### ❓ Problem Statement"
    )

    st.markdown(
        """
        <div class="about">

        Mental-health challenges can influence an employee's workplace
        experience, productivity and willingness to seek professional
        support. However, organizations may not always know which
        workplace and personal factors are associated with treatment
        seeking.

        <br><br>

        This project analyzes the OSMI survey to identify patterns
        between mental-health treatment and variables such as family
        history, workplace interference, employer benefits, care
        availability, anonymity and leave policies.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # BUSINESS OBJECTIVE
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Business Objective"
    )

    st.markdown(
        """
        <div class="insight">

        The business objective is to provide technology organizations
        and HR teams with data-driven insights into factors associated
        with mental-health treatment-seeking.

        <br><br>

        The analysis can help identify potential gaps in awareness,
        accessibility and workplace support so organizations can
        investigate appropriate employee-support strategies.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown(
        "### 💡 Recommendations"
    )

    recommendations = [
        (
            "Improve awareness of support resources",
            "Organizations should make existing mental-health benefits "
            "and support programs easy to find and understand."
        ),
        (
            "Communicate privacy policies",
            "Employees should receive clear information about "
            "confidentiality and anonymity when accessing support."
        ),
        (
            "Simplify mental-health leave information",
            "Employees should be able to understand available leave "
            "options without unnecessary administrative complexity."
        ),
        (
            "Strengthen manager awareness",
            "Managers can receive appropriate guidance about directing "
            "employees toward available resources."
        ),
        (
            "Monitor employee feedback",
            "Anonymous employee surveys can be repeated periodically "
            "to monitor changes in awareness and workplace experience."
        ),
    ]

    for i, (title, body) in enumerate(
        recommendations,
        1
    ):

        st.markdown(
            f"""
            <div class="recommendation">

                <div class="recommendation-title">
                    {i}. {title}
                </div>

                <div class="recommendation-body">
                    {body}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        "### ⚠️ Limitations"
    )

    st.markdown(
        """
        <div class="warning">

        • The dataset is based on voluntary survey participation.<br><br>

        • The respondents may not represent the entire technology
        workforce.<br><br>

        • The survey represents the context of 2014 and should not
        automatically be treated as a current population estimate.<br><br>

        • Small subgroups should be interpreted carefully.<br><br>

        • Observed associations do not establish causation.<br><br>

        • The analysis should not be used for clinical diagnosis or
        individual mental-health decisions.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    st.markdown(
        "### 🏁 Conclusion"
    )

    st.markdown(
        """
        <div class="about">

        The exploratory analysis of the OSMI Mental Health in Tech Survey
        demonstrates how employee survey data can be used to investigate
        patterns related to mental-health treatment and workplace
        experience.

        <br><br>

        The analysis highlights differences across family history,
        workplace interference, benefits, care options, anonymity,
        leave and other workplace variables.

        <br><br>

        These findings provide a foundation for further statistical
        analysis and machine-learning work while also demonstrating the
        importance of careful data cleaning, visualization and
        interpretation.

        <br><br>

        The main value of the project is not simply identifying a single
        factor associated with treatment. Instead, it demonstrates how
        multiple dimensions of workplace experience can be analyzed
        together to better understand the survey population.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # AUTHOR
    # --------------------------------------------------------

    st.markdown(
        "### 👩‍💻 About"
    )

    st.markdown(
        """
        <div class="about"
             style="text-align:center;">

            <div style="font-size:3rem;">
                🧠
            </div>

            <div style="
                font-size:1.8rem;
                font-weight:800;
                color:#20504A;
            ">
                Challapalli Mamatha
            </div>

            <div style="
                color:#647270;
                margin-top:0.5rem;
            ">
                Data Analytics & AI
            </div>

            <br>

            Mental Health in Tech Survey — EDA Capstone Project

        </div>
        """,
        unsafe_allow_html=True
    )