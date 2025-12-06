# Mental_Health_Dashboard.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------
# 1. Load & clean the data
# ---------------------------

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("survey.csv")

    # Basic types
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Clean Gender
    df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()

    female_terms = [
        "female (cis)", "female (cis)\t", "fem", "f", "femal", "woman", "femail",
        "cis female", "femake", "cis female", "female (cis)", "woman",
        "queer/she/they", "femake", "cis-female/femme"
    ]

    male_terms = [
        "msle", "guy (-ish) ^_^", "man", "mal", "make", "male (cis)", "male",
        "m", "cis man", "male,m", "male-ish", "malr", "maile", "mail",
        "cis male", "male-ish", "something kinda male?"
    ]

    other_terms = [
        "male leaning androgynous", "male leaning androgyous", "neuter",
        "trans-female", "unsure what that really means", "trans woman", "p",
        "genderqueer", "a little about you", "non-binary", "nah", "all", "enby",
        "trans woman", "neuter", "female (trans)", "queer", "fluid",
        "genderqueer", "female (trans)", "androgyne", "agender",
        "ostensibly male, unsure what that really means"
    ]

    df["Gender"] = df["Gender"].replace(female_terms, "female")
    df["Gender"] = df["Gender"].replace(male_terms, "male")
    df["Gender"] = df["Gender"].replace(other_terms, "other")

    # Replace general "unknown" values with NaN
    df.replace(
        ["", "N/A", "n/a", "Na", "Don't know", "Maybe", "Some of them"],
        np.nan,
        inplace=True,
    )

    # Valid age range
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

    # Treatment to bool
    df["treatment"] = df["treatment"].replace({"Yes": True, "No": False})
    df["treatment"] = df["treatment"].astype(bool)

    # Strip strings
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Clean no_employees
    df["no_employees"] = (
        df["no_employees"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"jun-25": np.nan, "01-may": np.nan})
    )

    # Cast to categories
    cat_cols = [
        "Gender", "no_employees", "Country", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor",
        "mental_health_interview", "phys_health_interview",
        "mental_vs_physical", "obs_consequence", "work_interfere",
        "benefits", "care_options", "leave"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Boolean columns
    bool_cols = [
        "remote_work", "tech_company", "seek_help", "self_employed",
        "family_history"
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype("bool")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Company size numeric mapping for scatter
    size_map = {
        "1-5": 3,
        "6-25": 15,
        "26-100": 63,
        "100-500": 300,
        "500-1000": 750,
        "more than 1000": 1200,
        "more than 1000 ": 1200,
        "More than 1000": 1200,
    }
    df["company_size"] = df["no_employees"].map(size_map)

    return df


df = load_and_clean_data()

# ---------------------------
# 2. Streamlit layout
# ---------------------------

st.set_page_config(
    page_title="Mental Health in Tech – Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health in Tech – Survey Dashboard")
st.write("Interactive dashboard based on your mid-project analysis.")

# ---------------------------
# 3. Sidebar filters
# ---------------------------

st.sidebar.header("Filters")

# Gender filter
genders = df["Gender"].dropna().unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", options=genders, default=genders)

# Age filter
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider("Age range", min_age, max_age, (min_age, max_age))

# Remote work filter
remote_options = ["All", "Remote", "On-site"]
remote_choice = st.sidebar.radio("Remote work", remote_options, index=0)

# Tech company filter
tech_options = ["All", "Tech only", "Non-tech only"]
tech_choice = st.sidebar.radio("Company type", tech_options, index=0)

# Country filter
countries = sorted(df["Country"].dropna().unique().tolist())
default_countries = countries[:10] if len(countries) > 10 else countries
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=default_countries,
)

# ---------------------------
# 4. Apply filters
# ---------------------------

filtered_df = df.copy()
filtered_df = filtered_df[
    filtered_df["Gender"].isin(selected_gender) &
    filtered_df["Age"].between(age_range[0], age_range[1])
]

if selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if remote_choice != "All":
    is_remote = (remote_choice == "Remote")
    filtered_df = filtered_df[filtered_df["remote_work"] == is_remote]

if tech_choice != "All":
    is_tech = (tech_choice == "Tech only")
    filtered_df = filtered_df[filtered_df["tech_company"] == is_tech]

# 👉 IMPORTANT: avoid errors when filters give 0 rows
if filtered_df.empty:
    st.warning(
        "No respondents match the current filters. "
        "Try selecting more countries or changing the filters."
    )
    st.stop()

st.caption(f"Showing **{len(filtered_df)}** respondents after filters.")

# ---------------------------
# 5. KPI cards
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total respondents", len(filtered_df))

with col2:
    if "treatment" in filtered_df.columns:
        treatment_rate = filtered_df["treatment"].mean() * 100
        st.metric("Treatment rate", f"{treatment_rate:.1f}%")
    else:
        st.metric("Treatment rate", "N/A")

with col3:
    if "family_history" in filtered_df.columns:
        fam_rate = filtered_df["family_history"].mean() * 100
        st.metric("Family history (Yes)", f"{fam_rate:.1f}%")
    else:
        st.metric("Family history", "N/A")

# ---------------------------
# 6. Tabs
# ---------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "🧠 Mental Health & Treatment",
    "🏢 Workplace Factors",
    "🌍 Country Comparison"
])

# -------- Tab 1: Overview --------
with tab1:
    st.subheader("Gender distribution")

    gender_counts = (
        filtered_df["Gender"]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "Gender", "Gender": "Count"})
    )

    fig_gender_pie = px.pie(
        gender_counts,
        names="Gender",
        values="Count",
        title="Gender distribution",
        hole=0.4
    )
    st.plotly_chart(fig_gender_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Age distribution by gender")

    if "Gender" in filtered_df.columns:
        fig_age_gender = px.histogram(
            filtered_df,
            x="Age",
            color="Gender",
            barmode="overlay",
            title="Age distribution by gender",
        )
        st.plotly_chart(fig_age_gender, use_container_width=True)
    else:
        st.info("No gender information available for this filtered subset.")

# -------- Tab 2: Mental Health & Treatment --------
with tab2:
    st.subheader("Mental-health consequences by gender")

    if "mental_health_consequence" in filtered_df.columns:
        fig_mhc_gender = px.histogram(
            filtered_df,
            x="Gender",
            color="mental_health_consequence",
            barmode="group",
            title="Mental health consequences by gender",
        )
        st.plotly_chart(fig_mhc_gender, use_container_width=True)
    else:
        st.info("No mental health consequence data available.")

    st.markdown("---")
    st.subheader("Remote work & mental-health consequences")

    if "remote_work" in filtered_df.columns and \
       "mental_health_consequence" in filtered_df.columns:
        fig_remote = px.histogram(
            filtered_df,
            x="remote_work",
            color="mental_health_consequence",
            barmode="group",
            title="Remote work and mental health consequences",
        )
        st.plotly_chart(fig_remote, use_container_width=True)

    st.markdown("---")
    st.subheader("Treatment rate by gender")

    if "treatment" in filtered_df.columns:
        treatment_rate_by_gender = (
            filtered_df
            .groupby("Gender", observed=False)["treatment"]
            .mean()
            .reset_index(name="treatment_rate")
        )

        fig_trt_gender = px.bar(
            treatment_rate_by_gender,
            x="Gender",
            y="treatment_rate",
            color="Gender",
            text="treatment_rate",
            title="Treatment rate by gender",
        )
        fig_trt_gender.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_trt_gender.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig_trt_gender, use_container_width=True)

# -------- Tab 3: Workplace Factors --------
with tab3:
    st.subheader("Treatment by company size (%)")

    if "no_employees" in filtered_df.columns and "treatment" in filtered_df.columns:
        employee_order = [
            "1-5", "6-25", "26-100",
            "100-500", "500-1000", "more than 1000", "more than 1000 "
        ]
        filtered_df["no_employees"] = pd.Categorical(
            filtered_df["no_employees"], categories=employee_order, ordered=True
        )

        fig_treatment_size = px.histogram(
            filtered_df,
            x="no_employees",
            color="treatment",
            barmode="stack",
            histnorm="percent",
            category_orders={"no_employees": employee_order},
            title="Treatment rate by company size (%)",
        )
        fig_treatment_size.update_layout(
            yaxis_title="Percentage",
            xaxis_title="Number of employees"
        )
        st.plotly_chart(fig_treatment_size, use_container_width=True)

    st.markdown("---")
    st.subheader("Work interference vs company size")

    if "work_interfere" in filtered_df.columns:
        tmp = filtered_df.dropna(subset=["no_employees", "work_interfere"])
        if len(tmp) > 0:
            fig_heat = px.density_heatmap(
                tmp,
                x="no_employees",
                y="work_interfere",
                title="Company size vs work interference (count heatmap)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No data after filtering for company size and work interference.")

    st.markdown("---")
    st.subheader("Tech company status vs work interference")

    if "tech_company" in filtered_df.columns and "work_interfere" in filtered_df.columns:
        tmp2 = (
            filtered_df
            .dropna(subset=["tech_company", "work_interfere"])
            .groupby(["tech_company", "work_interfere"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        if len(tmp2) > 0:
            fig_tech_work = px.bar(
                tmp2,
                x="tech_company",
                y="Count",
                color="work_interfere",
                barmode="group",
                title="Tech company status vs work interference",
            )
            st.plotly_chart(fig_tech_work, use_container_width=True)

# -------- Tab 4: Country Comparison --------
with tab4:
    st.subheader("Countries with highest reported mental health consequences (Yes)")

    if "mental_health_consequence" in filtered_df.columns:
        tmp = (
            filtered_df[filtered_df["mental_health_consequence"].eq("Yes")]
            .groupby("Country", as_index=False)
            .size()
            .rename(columns={"size": "Count"})
            .sort_values("Count", ascending=False)
            .head(15)
        )
        if len(tmp) > 0:
            fig_country_yes = px.bar(
                tmp.sort_values("Count"),
                x="Count",
                y="Country",
                orientation="h",
                title="Countries with highest reported mental health consequences (Yes)",
            )
            st.plotly_chart(fig_country_yes, use_container_width=True)

    st.markdown("---")
    st.subheader("Belief that mental and physical health are equally important")

    if "mental_vs_physical" in filtered_df.columns:
        counts = (
            filtered_df
            .dropna(subset=["Country", "mental_vs_physical"])
            .groupby(["Country", "mental_vs_physical"])
            .size()
            .reset_index(name="n")
        )
        if len(counts) > 0:
            total = counts.groupby("Country", as_index=False)["n"].sum().rename(columns={"n": "N"})
            yes = counts[counts["mental_vs_physical"].eq("Yes")][["Country", "n"]].rename(columns={"n": "Yes"})
            perc = total.merge(yes, on="Country", how="left").fillna({"Yes": 0})
            perc["Pct_Yes"] = 100 * perc["Yes"] / perc["N"]
            top = perc.sort_values("Pct_Yes", ascending=False).head(15)

            fig_pct_yes = px.bar(
                top,
                x="Country",
                y="Pct_Yes",
                title="Countries with highest % 'Yes' on mental vs physical health",
                text="Pct_Yes",
            )
            fig_pct_yes.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_pct_yes.update_layout(
                xaxis_title="Country",
                yaxis_title="% Yes",
            )
            st.plotly_chart(fig_pct_yes, use_container_width=True)
