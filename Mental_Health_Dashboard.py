import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------
# 1. Load & clean data
# --------------------------------------

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("survey.csv")

    # Basic types
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Clean Gender
    df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()

    female_terms = [
        "female (cis)", "female (cis)\t", "fem", "f", "femal",
        "woman", "femail", "cis female", "femake",
        "female (cis)", "queer/she/they", "cis-female/femme"
    ]

    male_terms = [
        "msle", "guy (-ish) ^_^", "man", "mal", "make",
        "male (cis)", "male", "m", "male,m",
        "male-ish", "malr", "maile", "mail", "cis male",
        "something kinda male?"
    ]

    other_terms = [
        "male leaning androgynous", "male leaning androgyous", "neuter",
        "trans-female", "unsure what that really means", "trans woman", "p",
        "genderqueer", "a little about you", "non-binary", "nah", "all",
        "enby", "female (trans)", "queer", "fluid", "androgyne", "agender",
        "ostensibly male, unsure what that really means"
    ]

    df["Gender"] = df["Gender"].replace(female_terms, "female")
    df["Gender"] = df["Gender"].replace(male_terms, "male")
    df["Gender"] = df["Gender"].replace(other_terms, "other")

    # Replace generic missing values
    df.replace(
        ["", "N/A", "n/a", "Na", "Don't know", "Maybe", "Some of them"],
        np.nan,
        inplace=True,
    )

    # Keep valid age range
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

    # Category columns
    cat_cols = [
        "Gender", "no_employees", "Country", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor",
        "mental_health_interview", "phys_health_interview",
        "mental_vs_physical", "obs_consequence", "work_interfere",
        "benefits", "care_options", "leave", "anonymity"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Boolean columns
    bool_cols = [
        "remote_work", "tech_company", "seek_help",
        "self_employed", "family_history"
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype("bool")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Company size numeric mapping
    size_map = {
        "1-5": 3,
        "6-25": 15,
        "26-100": 63,
        "100-500": 300,
        "500-1000": 750,
        "more than 1000": 1200,
        "More than 1000": 1200,
    }
    df["company_size"] = df["no_employees"].map(size_map)

    return df


df = load_and_clean_data()

# --------------------------------------
# 2. Page config & title
# --------------------------------------

st.set_page_config(
    page_title="Mental Health in Tech – Survey Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health in Tech – Survey Dashboard")
st.write("Interactive dashboard based on your mid-project analysis.")

# --------------------------------------
# 3. Sidebar filters
# --------------------------------------

st.sidebar.header("Filters")

# Gender
genders = df["Gender"].dropna().unique().tolist()
selected_gender = st.sidebar.multiselect(
    "Gender",
    options=genders,
    default=genders
)

# Age range
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider(
    "Age range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Remote work
remote_options = ["All", "Remote", "On-site"]
remote_choice = st.sidebar.radio("Remote work", remote_options, index=0)

# Company type
tech_options = ["All", "Tech only", "Non-tech only"]
tech_choice = st.sidebar.radio("Company type", tech_options, index=0)

# Country filter
countries = sorted(df["Country"].dropna().unique().tolist())
default_countries = countries[:10] if len(countries) > 10 else countries
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=default_countries
)

# --------------------------------------
# 4. Apply filters
# --------------------------------------

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

# Avoid crash when there is no data
if filtered_df.empty:
    st.warning(
        "No respondents match the current filters. "
        "Try selecting more countries or relaxing the filters."
    )
    st.stop()

st.caption(f"Showing **{len(filtered_df)}** respondents after filters.")

# --------------------------------------
# 5. KPI cards
# --------------------------------------

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

# --------------------------------------
# 6. Tabs
# --------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "🧠 Mental Health & Treatment",
    "🏢 Workplace Factors",
    "🌍 Country Comparison"
])

# ============================================================
# TAB 1 : OVERVIEW  (2 charts)
# ============================================================
with tab1:
    st.subheader("Gender distribution")

    gender_counts = filtered_df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

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

    fig_age_gender = px.histogram(
        filtered_df,
        x="Age",
        color="Gender",
        barmode="overlay",
        title="Age distribution by gender",
    )
    st.plotly_chart(fig_age_gender, use_container_width=True)

# ============================================================
# TAB 2 : MENTAL HEALTH & TREATMENT  (8 charts)
# ============================================================
with tab2:
    # 1) Mental-health consequences by gender
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

    st.markdown("---")

    # 2) Remote work & mental-health consequences
    st.subheader("Remote work & mental-health consequences")

    if "remote_work" in filtered_df.columns:
        fig_remote = px.histogram(
            filtered_df,
            x="remote_work",
            color="mental_health_consequence",
            barmode="group",
            title="Remote work and mental health consequences",
        )
        st.plotly_chart(fig_remote, use_container_width=True)

    st.markdown("---")

    # 3) Age vs mental health consequences by gender (violin)
    st.subheader("Age vs mental-health consequences by gender")

    if "mental_health_consequence" in filtered_df.columns:
        d = filtered_df[["Age", "mental_health_consequence", "Gender"]].dropna()
        if not d.empty:
            order = ["No", "Maybe", "Yes"]
            fig_age_mhc = px.violin(
                d,
                x="mental_health_consequence",
                y="Age",
                color="Gender",
                box=True,
                points="all",
                category_orders={"mental_health_consequence": order},
                title="Age vs mental health consequences by gender",
            )
            fig_age_mhc.update_layout(
                xaxis_title="Mental health consequence",
                yaxis_title="Age"
            )
            st.plotly_chart(fig_age_mhc, use_container_width=True)

    st.markdown("---")

    # 4) Treatment rate by gender
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
        fig_trt_gender.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
        )
        fig_trt_gender.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig_trt_gender, use_container_width=True)

    st.markdown("---")

    # 5) Family history by gender
    st.subheader("Family history of mental illness by gender")

    if "family_history" in filtered_df.columns:
        tmp_fh = filtered_df.dropna(subset=["Gender", "family_history"])
        if not tmp_fh.empty:
            fig_fam_hist = px.histogram(
                tmp_fh,
                x="Gender",
                color="family_history",
                barmode="stack",
                title="Family history of mental illness by gender",
            )
            st.plotly_chart(fig_fam_hist, use_container_width=True)

    st.markdown("---")

    # 6) Age vs company size vs treatment (scatter)
    st.subheader("Age vs company size (coloured by treatment)")

    if "company_size" in filtered_df.columns:
        fig_age_comp = px.scatter(
            filtered_df,
            x="Age",
            y="company_size",
            color="treatment",
            title="Age vs company size (coloured by treatment)",
            labels={"company_size": "Approximate company size"},
        )
        st.plotly_chart(fig_age_comp, use_container_width=True)

# ============================================================
# TAB 3 : WORKPLACE FACTORS  (11 charts)
# ============================================================
with tab3:
    # 1) Treatment by company size (%)
    st.subheader("Treatment rate by company size (%)")

    if "no_employees" in filtered_df.columns and "treatment" in filtered_df.columns:
        employee_order = [
            "1-5", "6-25", "26-100",
            "100-500", "500-1000",
            "more than 1000", "More than 1000"
        ]

        filtered_df["no_employees"] = pd.Categorical(
            filtered_df["no_employees"],
            categories=employee_order,
            ordered=True
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

    # 2) Company size vs work interference (heatmap)
    st.subheader("Company size vs work interference")

    if "work_interfere" in filtered_df.columns:
        tmp_wi = filtered_df.dropna(subset=["no_employees", "work_interfere"])
        if not tmp_wi.empty:
            fig_heat = px.density_heatmap(
                tmp_wi,
                x="no_employees",
                y="work_interfere",
                title="Company size vs work interference (count heatmap)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # 3) Age vs work interference (violin, coloured by treatment)
    st.subheader("Age vs work interference (by treatment)")

    if "work_interfere" in filtered_df.columns:
        tmp_vi = filtered_df.dropna(subset=["Age", "work_interfere", "treatment"])
        if not tmp_vi.empty:
            fig_vi_work = px.violin(
                tmp_vi,
                x="work_interfere",
                y="Age",
                color="treatment",
                box=True,
                points="all",
                category_orders={
                    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"]
                },
                title="Age vs work interference (coloured by treatment)",
            )
            fig_vi_work.update_layout(
                xaxis_title="Work interference",
                yaxis_title="Age"
            )
            st.plotly_chart(fig_vi_work, use_container_width=True)

    st.markdown("---")

    # 4) Tech company status vs work interference
    st.subheader("Tech company status vs work interference")

    if "tech_company" in filtered_df.columns and "work_interfere" in filtered_df.columns:
        tmp2 = (
            filtered_df
            .dropna(subset=["tech_company", "work_interfere"])
            .groupby(["tech_company", "work_interfere"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        if not tmp2.empty:
            fig_tech_work = px.bar(
                tmp2,
                x="tech_company",
                y="Count",
                color="work_interfere",
                barmode="group",
                title="Tech company status vs work interference",
            )
            st.plotly_chart(fig_tech_work, use_container_width=True)

    st.markdown("---")

    # 5) Treatment by country & company size (treemap)
    st.subheader("Mental-health treatment by country & company size")

    temp_df = filtered_df.copy()
    temp_df["no_employees_treemap"] = (
        temp_df["no_employees"]
        .astype(str)
        .replace({"nan": "Not specified"})
    )
    fig_treemap = px.treemap(
        temp_df,
        path=["Country", "no_employees_treemap", "treatment"],
        title="Mental health treatment by country & company size",
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    st.markdown("---")

    # 6) Supervisor support vs treatment
    st.subheader("Supervisor support vs treatment")

    if "supervisor" in filtered_df.columns:
        tmp_sup = (
            filtered_df
            .dropna(subset=["supervisor", "treatment"])
            .groupby(["supervisor", "treatment"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        if not tmp_sup.empty:
            fig_sup = px.bar(
                tmp_sup,
                x="supervisor",
                y="Count",
                color="treatment",
                barmode="group",
                title="Supervisor support vs treatment",
            )
            st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("---")

    # 7) Benefits by tech company status
    st.subheader("Benefits by tech company status")

    if "benefits" in filtered_df.columns and "tech_company" in filtered_df.columns:
        tmp_ben = filtered_df.dropna(subset=["tech_company", "benefits"])
        if not tmp_ben.empty:
            fig_ben_tech = px.histogram(
                tmp_ben,
                x="tech_company",
                color="benefits",
                barmode="stack",
                title="Benefits by tech company status",
            )
            st.plotly_chart(fig_ben_tech, use_container_width=True)

    st.markdown("---")

    # 8) Benefits vs willingness to seek help
    st.subheader("Benefits vs willingness to seek help")

    if "benefits" in filtered_df.columns and "seek_help" in filtered_df.columns:
        tmp_seek = filtered_df.dropna(subset=["benefits", "seek_help"])
        if not tmp_seek.empty:
            fig_seek = px.histogram(
                tmp_seek,
                x="benefits",
                color="seek_help",
                barmode="stack",
                title="Benefits vs willingness to seek professional help",
            )
            st.plotly_chart(fig_seek, use_container_width=True)

    st.markdown("---")

    # 9) Anonymity policy vs ease of taking leave
    st.subheader("Anonymity policy vs ease of taking leave")

    if "anonymity" in filtered_df.columns and "leave" in filtered_df.columns:
        ct = (
            filtered_df
            .dropna(subset=["anonymity", "leave"])
            .groupby(["anonymity", "leave"])
            .size()
            .reset_index(name="Count")
        )
        if not ct.empty:
            fig_anon = px.density_heatmap(
                ct,
                x="anonymity",
                y="leave",
                z="Count",
                title="Anonymity policy vs ease of taking leave (count heatmap)",
            )
            st.plotly_chart(fig_anon, use_container_width=True)

    st.markdown("---")

    # 10) Benefits availability (donut chart)
    st.subheader("Benefits availability")

    if "benefits" in filtered_df.columns:
        tmp_ben2 = filtered_df.dropna(subset=["benefits"])
        if not tmp_ben2.empty:
            fig_ben_pie = px.pie(
                tmp_ben2,
                names="benefits",
                title="Benefits availability",
                hole=0.4,
            )
            st.plotly_chart(fig_ben_pie, use_container_width=True)

# ============================================================
# TAB 4 : COUNTRY COMPARISON  (2 charts)
# ============================================================
with tab4:
    # 1) Countries with highest reported Yes consequences
    st.subheader("Countries with highest reported mental-health consequences (Yes)")

    if "mental_health_consequence" in filtered_df.columns:
        tmp = (
            filtered_df[filtered_df["mental_health_consequence"].eq("Yes")]
            .groupby("Country", as_index=False)
            .size()
            .rename(columns={"size": "Count"})
            .sort_values("Count", ascending=False)
            .head(15)
        )
        if not tmp.empty:
            fig_country_yes = px.bar(
                tmp.sort_values("Count"),
                x="Count",
                y="Country",
                orientation="h",
                title="Countries with highest reported mental health consequences (Yes)",
            )
            st.plotly_chart(fig_country_yes, use_container_width=True)

    st.markdown("---")

    # 2) Belief that mental & physical health are equally important
    st.subheader("Belief that mental & physical health are equally important")

    if "mental_vs_physical" in filtered_df.columns:
        counts = (
            filtered_df
            .dropna(subset=["Country", "mental_vs_physical"])
            .groupby(["Country", "mental_vs_physical"])
            .size()
            .reset_index(name="n")
        )
        if not counts.empty:
            total = (
                counts
                .groupby("Country", as_index=False)["n"]
                .sum()
                .rename(columns={"n": "N"})
            )
            yes = (
                counts[counts["mental_vs_physical"].eq("Yes")]
                [["Country", "n"]]
                .rename(columns={"n": "Yes"})
            )
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
            fig_pct_yes.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            fig_pct_yes.update_layout(
                xaxis_title="Country",
                yaxis_title="% Yes",
            )
            st.plotly_chart(fig_pct_yes, use_container_width=True)
