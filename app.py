import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Online Food Analytics Dashboard",
    page_icon="🍔",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

h1,h2,h3 {
    color: #1e293b;
}

[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD DATA
# ---------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("onlinefoods.csv")
    return df

df = load_data()

# Remove unwanted column
if "Unnamed: 12" in df.columns:
    df.drop("Unnamed: 12", axis=1, inplace=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("🍕 Food Ordering Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Demographics",
        "Income Analysis",
        "Occupation Analysis",
        "Customer Feedback",
        "Location Analysis",
        "Prediction"
    ]
)

# ---------------------------------
# DASHBOARD
# ---------------------------------

if page == "Dashboard":

    st.title("🍔 Online Food Ordering Analytics")

    total_customers = len(df)

    positive_feedback = round(
        (df["Feedback"].str.strip() == "Positive").mean()*100,
        1
    )

    order_rate = round(
        (df["Output"] == "Yes").mean()*100,
        1
    )

    avg_age = round(df["Age"].mean(),1)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Customers", total_customers)
    c2.metric("Average Age", avg_age)
    c3.metric("Positive Feedback %", positive_feedback)
    c4.metric("Order Intention %", order_rate)

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Age",
            title="Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            df,
            names="Gender",
            title="Gender Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ---------------------------------
# DEMOGRAPHICS
# ---------------------------------

elif page == "Demographics":

    st.title("👥 Demographic Analysis")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Age",
            title="Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            df["Marital Status"].value_counts().reset_index(),
            x="Marital Status",
            y="count",
            title="Marital Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    edu = df["Educational Qualifications"].value_counts().reset_index()

    fig = px.bar(
        edu,
        x="Educational Qualifications",
        y="count",
        title="Educational Qualification Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------
# INCOME ANALYSIS
# ---------------------------------

elif page == "Income Analysis":

    st.title("💰 Income Analysis")

    income_counts = (
        df["Monthly Income"]
        .value_counts()
        .reset_index()
    )

    fig = px.bar(
        income_counts,
        x="Monthly Income",
        y="count",
        title="Income Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    income_order = pd.crosstab(
        df["Monthly Income"],
        df["Output"]
    )

    st.subheader("Income vs Ordering")

    st.dataframe(income_order)

# ---------------------------------
# OCCUPATION ANALYSIS
# ---------------------------------

elif page == "Occupation Analysis":

    st.title("💼 Occupation Analysis")

    occ = (
        df["Occupation"]
        .value_counts()
        .reset_index()
    )

    fig = px.bar(
        occ,
        x="Occupation",
        y="count",
        title="Occupation Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.sunburst(
        df,
        path=[
            "Occupation",
            "Output"
        ],
        title="Occupation vs Ordering"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------
# FEEDBACK ANALYSIS
# ---------------------------------

elif page == "Customer Feedback":

    st.title("⭐ Customer Feedback")

    fig = px.pie(
        df,
        names="Feedback",
        title="Feedback Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    feedback_gender = pd.crosstab(
        df["Gender"],
        df["Feedback"]
    )

    st.subheader("Gender vs Feedback")

    st.dataframe(feedback_gender)

# ---------------------------------
# LOCATION ANALYSIS
# ---------------------------------

elif page == "Location Analysis":

    st.title("📍 Customer Location Analysis")

    st.map(
        df[
            ["latitude","longitude"]
        ]
    )

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="Occupation",
        zoom=10
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------
# MACHINE LEARNING
# ---------------------------------

elif page == "Prediction":

    st.title("🤖 Food Ordering Prediction")

    model_df = df.copy()

    encoders = {}

    cols = [
        "Gender",
        "Marital Status",
        "Occupation",
        "Monthly Income",
        "Educational Qualifications",
        "Output"
    ]

    for col in cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le

    X = model_df[
        [
            "Age",
            "Gender",
            "Marital Status",
            "Occupation",
            "Monthly Income",
            "Family size"
        ]
    ]

    y = model_df["Output"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X,y)

    st.subheader("Enter Customer Details")

    age = st.slider(
        "Age",
        18,
        60,
        25
    )

    gender = st.selectbox(
        "Gender",
        df["Gender"].unique()
    )

    marital = st.selectbox(
        "Marital Status",
        df["Marital Status"].unique()
    )

    occupation = st.selectbox(
        "Occupation",
        df["Occupation"].unique()
    )

    income = st.selectbox(
        "Monthly Income",
        df["Monthly Income"].unique()
    )

    family = st.slider(
        "Family Size",
        1,
        10,
        4
    )

    if st.button("Predict"):

        input_df = pd.DataFrame({
            "Age":[age],
            "Gender":[encoders["Gender"].transform([gender])[0]],
            "Marital Status":[encoders["Marital Status"].transform([marital])[0]],
            "Occupation":[encoders["Occupation"].transform([occupation])[0]],
            "Monthly Income":[encoders["Monthly Income"].transform([income])[0]],
            "Family size":[family]
        })

        pred = model.predict(input_df)[0]

        result = encoders["Output"].inverse_transform([pred])[0]

        if result == "Yes":
            st.success(
                "Customer is likely to order food online ✅"
            )
        else:
            st.error(
                "Customer is unlikely to order food online ❌"
            )

# ---------------------------------
# FOOTER
# ---------------------------------

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit")
