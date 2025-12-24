import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests



from src.config import URL

st.set_page_config(page_title="Job Finder", layout="wide")
st.title("🔍 Job Finder")


DEFAULT_STATE = {
    "step": 1,
    "domain": None,
    "Tastes": None,
    "Questions": None,
    "Skills": None,
    "Jobs_competencies": None,
    "Skills_competency": None,
}

for k, v in DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)


def reset_app():
    for k in DEFAULT_STATE:
        st.session_state[k] = DEFAULT_STATE[k]
    st.rerun()



@st.cache_data(show_spinner=False)
def load_reference_data():
    r = requests.get(f"{URL}/recommender/get-df")
    r.raise_for_status()
    data = r.json()
    return (
        pd.DataFrame(data["tastes"]),
        pd.DataFrame(data["questions"]),
        pd.DataFrame(data["skills"]),
    )


try:
    st.session_state["Tastes"], st.session_state["Questions"], st.session_state["Skills"] = load_reference_data()
except Exception as e:
    st.error(f"API error while loading data: {e}")
    st.stop()



if st.session_state["step"] == 1:
    st.header("Tell us about your preferences")

    with st.form("preferences_form"):
        user_input1 = st.text_area("✅ What do you like?", height=150)
        user_input2 = st.text_area("❌ What do you dislike?", height=150)

        submitted = st.form_submit_button("Continue to Domain Analysis", type="primary")

    if submitted:
        if not user_input1.strip() or not user_input2.strip():
            st.warning("Please fill in both fields.")
        else:
            with st.spinner("Analyzing your preferences..."):
                r = requests.post(
                    f"{URL}/recommender/get-domain",
                    params={"input1": user_input1, "input2": user_input2},
                )
                r.raise_for_status()
                data = r.json()

                st.session_state["Tastes"] = pd.DataFrame(data["tastes_df"])
                st.session_state["domain"] = data["domain"]
                st.session_state["step"] = 2
                st.rerun()



elif st.session_state["step"] == 2:
    domain = st.session_state["domain"]
    st.success(f"Best matching domain: **{domain}**")

    skills_domain = st.session_state["Skills"].query("Domain == @domain")

    if skills_domain.empty:
        st.error(f"No skills found for domain: {domain}")
        if st.button("🔄 Start Over"):
            reset_app()
    else:
        question = (
            st.session_state["Questions"]
            .query("Domain == @domain")["Questions"]
            .iloc[0]
        )

        st.subheader(f"Your experience in {domain}")

        with st.form("skills_form"):
            user_input3 = st.text_area("💼 Your competencies", height=120)
            user_input4 = st.text_area("📂 Your projects", height=120)
            user_input5 = st.text_area(f"➕ {question}", height=120)

            submitted = st.form_submit_button("Analyze My Profile", type="primary")

        if submitted:
            if not all(map(str.strip, [user_input3, user_input4, user_input5])):
                st.warning("Please fill in all fields.")
            else:
                with st.spinner("Analyzing your profile..."):
                    r = requests.post(
                        f"{URL}/recommender/question-based-analysis",
                        params={
                            "domain": domain,
                            "input3": user_input3,
                            "input4": user_input4,
                            "input5": user_input5,
                        },
                    )
                    r.raise_for_status()
                    data = r.json()

                    st.session_state["Jobs_competencies"] = pd.DataFrame(data["jobs_competencies"])
                    st.session_state["Skills_competency"] = pd.DataFrame(data["skills_competency"])
                    st.session_state["step"] = 3
                    st.rerun()

    if st.button("🔄 Start New Search"):
        reset_app()


elif st.session_state["step"] == 3:
    st.header("📜 Your Results")

    st.subheader("🏆 Top 3 Matching Jobs")
    top3 = st.session_state["Jobs_competencies"].head(3)

    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.metric(
                label=f"{medals[i]} #{i+1}",
                value=row["Job"],
                delta=f"Score: {row['Score']:.3f}",
            )

    st.divider()


    st.subheader("💡 Top Competencies")
    top_skills = st.session_state["Skills_competency"].head(5)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Greens(top_skills["Weight"] / top_skills["Weight"].max())

    ax_bar.barh(top_skills["Competency"], top_skills["Score"], color=colors)
    ax_bar.set_xlabel("Score")
    ax_bar.set_title("Top 5 Competencies (weighted)")
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.3)

    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.subheader("🎯 Similarity Analysis")
    radar_data = top_skills[["Competency", "sim3", "sim4", "sim5"]]

    categories = ["Competencies", "Projects", "Specific Question"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(radar_data)))

    for idx, (_, row) in enumerate(radar_data.iterrows()):
        values = [row["sim3"], row["sim4"], row["sim5"]]
        values += values[:1]
        ax_radar.plot(angles, values, linewidth=2, label=row["Competency"], color=colors_radar[idx])
        ax_radar.fill(angles, values, alpha=0.2, color=colors_radar[idx])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Similarity Scores by Input Type", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    st.pyplot(fig_radar)
    plt.close(fig_radar)


    with st.expander("📋 Detailed Competency Scores"):
        st.dataframe(
            st.session_state["Skills_competency"],
            use_container_width=True
        )

    st.divider()

    if st.button("🔄 Start New Search", type="primary"):
        reset_app()
