import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Smart Campus Analytics", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Cleaned_Student_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
FEATURES = ["Gender", "Age", "College", "Department", "Semester", "Attendance_%", "CGPA", "Backlogs", "Internship", "Skills"]

st.title("🎓 Smart Campus Analytics")
st.subheader("Student Performance & Placement Insights")
st.caption("Machine Learning • Data Analysis • Interactive Dashboard")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found. Put Cleaned_Student_Dataset.csv beside app.py.")
    st.stop()

df = load_data(DATA_PATH)
required = set(FEATURES + ["Placement_Status"])
missing = required.difference(df.columns)
if missing:
    st.error("Missing dataset columns: " + ", ".join(sorted(missing)))
    st.stop()

with st.sidebar:
    st.header("Dashboard Filters")
    department_options = sorted(df["Department"].dropna().astype(str).unique())
    gender_options = sorted(df["Gender"].dropna().astype(str).unique())
    internship_options = sorted(df["Internship"].dropna().astype(str).unique())
    selected_departments = st.multiselect("Department", department_options, default=department_options)
    selected_genders = st.multiselect("Gender", gender_options, default=gender_options)
    selected_internships = st.multiselect("Internship", internship_options, default=internship_options)

data = df[
    df["Department"].astype(str).isin(selected_departments)
    & df["Gender"].astype(str).isin(selected_genders)
    & df["Internship"].astype(str).isin(selected_internships)
].copy()

if data.empty:
    st.warning("No records match the selected filters. Select at least one option in each filter.")
    st.stop()

status = data["Placement_Status"].astype(str).str.strip().str.lower()
placed_mask = status.eq("placed")
not_placed_mask = status.eq("not placed")
placed_count = int(placed_mask.sum())
not_placed_count = int(not_placed_mask.sum())
rate = placed_count / len(data) * 100
avg_cgpa = pd.to_numeric(data["CGPA"], errors="coerce").mean()
avg_attendance = pd.to_numeric(data["Attendance_%"], errors="coerce").mean()
avg_backlogs = pd.to_numeric(data["Backlogs"], errors="coerce").mean()
internship_rate = data["Internship"].astype(str).str.strip().str.lower().eq("yes").mean() * 100
placed_rows = data[placed_mask]
avg_package = pd.to_numeric(placed_rows["Package_LPA"], errors="coerce").mean() if "Package_LPA" in data.columns else float("nan")

st.markdown("### Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", f"{len(data):,}")
c2.metric("Placed", f"{placed_count:,}")
c3.metric("Not Placed", f"{not_placed_count:,}")
c4.metric("Placement Rate", f"{rate:.2f}%")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Average CGPA", f"{avg_cgpa:.2f}" if pd.notna(avg_cgpa) else "N/A")
c6.metric("Average Attendance", f"{avg_attendance:.2f}%" if pd.notna(avg_attendance) else "N/A")
c7.metric("Average Backlogs", f"{avg_backlogs:.2f}" if pd.notna(avg_backlogs) else "N/A")
c8.metric("Internship Rate", f"{internship_rate:.2f}%")
if pd.notna(avg_package):
    st.metric("Average Package (placed students)", f"{avg_package:.2f} LPA")

tab_dashboard, tab_prediction, tab_data, tab_about = st.tabs(["📊 Dashboard", "🤖 Placement Prediction", "🧾 Dataset", "ℹ️ About"])

with tab_dashboard:
    left, right = st.columns(2)
    with left:
        counts = data["Placement_Status"].value_counts().rename_axis("Placement Status").reset_index(name="Students")
        fig = px.bar(counts, x="Placement Status", y="Students", color="Placement Status", text="Students", title="Overall Placement Status")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dept_counts = data.groupby(["Department", "Placement_Status"]).size().reset_index(name="Students")
        fig = px.bar(dept_counts, x="Department", y="Students", color="Placement_Status", barmode="group", title="Department-wise Placement")
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.box(data, x="Placement_Status", y="CGPA", color="Placement_Status", title="CGPA vs Placement"), use_container_width=True)
    with right:
        st.plotly_chart(px.box(data, x="Placement_Status", y="Attendance_%", color="Placement_Status", title="Attendance vs Placement"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        backlog_counts = data.groupby(["Backlogs", "Placement_Status"]).size().reset_index(name="Students")
        st.plotly_chart(px.bar(backlog_counts, x="Backlogs", y="Students", color="Placement_Status", barmode="group", title="Backlogs vs Placement"), use_container_width=True)
    with right:
        internship_counts = data.groupby(["Internship", "Placement_Status"]).size().reset_index(name="Students")
        st.plotly_chart(px.bar(internship_counts, x="Internship", y="Students", color="Placement_Status", barmode="group", title="Internship vs Placement"), use_container_width=True)

    age_counts = data.groupby(["Age", "Placement_Status"]).size().reset_index(name="Students")
    st.plotly_chart(px.line(age_counts, x="Age", y="Students", color="Placement_Status", markers=True, title="Age-wise Placement"), use_container_width=True)
    skill_counts = data.groupby(["Skills", "Placement_Status"]).size().reset_index(name="Students")
    skill_fig = px.bar(skill_counts, x="Skills", y="Students", color="Placement_Status", barmode="group", title="Skills vs Placement")
    skill_fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(skill_fig, use_container_width=True)

with tab_prediction:
    st.subheader("Predict a student's placement status")
    st.info("The result is a model estimate, not a guarantee of actual placement.")
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        st.error("Model files not found. Put random_forest_model.pkl and preprocessor.pkl beside app.py.")
    else:
        try:
            model, preprocessor = load_artifacts()
            with st.form("prediction_form"):
                a, b = st.columns(2)
                with a:
                    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().astype(str).unique()))
                    age = st.number_input("Age", min_value=16, max_value=70, value=21)
                    college = st.selectbox("College", sorted(df["College"].dropna().astype(str).unique()))
                    department = st.selectbox("Department", sorted(df["Department"].dropna().astype(str).unique()))
                    semester = st.number_input("Semester", min_value=1, max_value=12, value=7)
                with b:
                    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
                    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
                    backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0)
                    internship = st.selectbox("Internship", sorted(df["Internship"].dropna().astype(str).unique()))
                    skills = st.selectbox("Skills", sorted(df["Skills"].dropna().astype(str).unique()))
                submitted = st.form_submit_button("Predict Placement", type="primary")
            if submitted:
                student = pd.DataFrame([{
                    "Gender": gender, "Age": age, "College": college, "Department": department,
                    "Semester": semester, "Attendance_%": attendance, "CGPA": cgpa,
                    "Backlogs": backlogs, "Internship": internship, "Skills": skills
                }], columns=FEATURES)
                transformed = preprocessor.transform(student)
                prediction = model.predict(transformed)[0]
                st.success(f"Predicted Placement Status: {prediction}")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(transformed)[0]
                    prob_df = pd.DataFrame({"Status": list(model.classes_), "Estimated Probability (%)": [float(p) * 100 for p in probs]})
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    st.plotly_chart(px.bar(prob_df, x="Status", y="Estimated Probability (%)", color="Status", title="Model-estimated probabilities"), use_container_width=True)
        except Exception as exc:
            st.error("Model loading or prediction failed. Ensure model and preprocessor were saved from the same training workflow.")
            st.exception(exc)

with tab_data:
    st.subheader("Student records")
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", data.to_csv(index=False).encode("utf-8"), "filtered_student_dataset.csv", "text/csv")

with tab_about:
    st.write("Smart Campus Analytics explores student performance and placement patterns using a cleaned dataset and a trained Random Forest model.")
    st.markdown("**Workflow**\n1. Data cleaning and preprocessing\n2. Exploratory data analysis\n3. Interactive visualization\n4. Model training and testing\n5. Dashboard and placement prediction")
    st.warning("Dataset placement rate and model test accuracy are different metrics. Predictions are estimates, not guarantees.")

st.caption("Smart Campus Analytics | Student Performance & Placement Insights")
