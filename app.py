import streamlit as st
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import unidecode
from pathlib import Path

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.remove("not")  # keep negations
stop_words.remove("no")   # keep negations

DATA_PATH = Path(r"data\data_project.xlsx")



def normalize(text: str):
    """
    Clean and format text for NLP processing.
    """
    text = unidecode.unidecode(text)  # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    text = text.lower()  # lowercase
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    text = " ".join(words)
    return text

# Initialize session state

if "step" not in st.session_state:
    st.session_state["step"] = 1  # Current phase of the app
if "domain" not in st.session_state:
    st.session_state["domain"] = None  # Best-matching domain inferred from user preferences
if "embedded_skills" not in st.session_state:
    st.session_state["embedded_skills"] = None  # Vector representations of skills for similarity matching
if "Tastes" not in st.session_state:
    st.session_state["Tastes"] = None  # Dataset containing user taste descriptions and domains
if "Skills_competency" not in st.session_state:
    st.session_state["Skills_competency"] = None  # Computed competency-skill-similarity results
if "user_input1" not in st.session_state:
    st.session_state["user_input1"] = ""  # User’s positive preferences
if "user_input2" not in st.session_state:
    st.session_state["user_input2"] = ""  # User’s dislikes or constraints
if "user_input3" not in st.session_state:
    st.session_state["user_input3"] = ""  # User’s competencies in the identified domain
if "user_input4" not in st.session_state:
    st.session_state["user_input4"] = ""  # User’s completed projects
if "user_input5" not in st.session_state:
    st.session_state["user_input5"] = ""  # User’s answer to the domain-specific question
if "model" not in st.session_state:
    # NLP model for embeddings
    st.session_state["model"] = SentenceTransformer("usc-isi/sbert-roberta-large-anli-mnli-snli")  

# Reset the app but keep model and caches
def reset_app():
    """
    Reset the Streamlit session state while keeping the loaded model and cached embeddings.
    The skill embeddings are preserved in cache to avoid recomputing them, improving performance
    when restarting or reanalyzing user inputs.
    """
    keys_to_keep = ["model", "embedded_tastes", "embedded_skills"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
      
    st.session_state["step"] = 1
    st.rerun()

st.title("🔍 Job Finder")

# Load data
try:
    if st.session_state["Tastes"] is None:
        st.session_state["Tastes"] = pd.read_excel(DATA_PATH, sheet_name=0)
    Questions = pd.read_excel(DATA_PATH, sheet_name=1)
    Skills = pd.read_excel(DATA_PATH, sheet_name=2)

    # Basic data validation
    if "Domain" not in st.session_state["Tastes"].columns:
        st.error("Error: 'Domain' column not found in Tastes sheet")
        st.stop()
    if "Domain" not in Questions.columns:
        st.error("Error: 'Domain' column not found in Questions sheet")
        st.stop()
    if "Domain" not in Skills.columns:
        st.error("Error: 'Domain' column not found in Skills sheet")
        st.stop()

except FileNotFoundError:
    st.error(f"Error: '{DATA_PATH}' file not found!")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Get user preferences
if st.session_state["step"] == 1:
    st.header("Tell us about your preferences")
    st.write("Describe what you like and don't like to help us find your ideal domain.")
    
    user_input1 = st.text_area("✅ Describe what you like:", height=150, value=st.session_state["user_input1"], key="input1_widget")
    user_input2 = st.text_area("❌ Describe what you don't like:", height=150, value=st.session_state["user_input2"], key="input2_widget")
    
    if st.button("Continue to Domain Analysis", type="primary", disabled=(user_input1.strip() == "" or user_input2.strip() == "")):
        st.session_state["user_input1"] = "I like " + user_input1
        st.session_state["user_input2"] = "I dont like "+ user_input2
        
        with st.spinner("Analyzing your preferences..."):
            # Encode tastes only once
            if "embedded_tastes" not in st.session_state:
                st.session_state["embedded_tastes"] = st.session_state["model"].encode( 
                    st.session_state["Tastes"]["Tastes"].apply(normalize), convert_to_tensor=True)
            
            # Clean and embed user inputs
            cleaned_user_input1 = normalize(st.session_state["user_input1"])
            cleaned_user_input2 = normalize(st.session_state["user_input2"])
            embedded_user1 = st.session_state["model"].encode(cleaned_user_input1, convert_to_tensor=True)
            embedded_user2 = st.session_state["model"].encode(cleaned_user_input2, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarities1 = util.cos_sim(embedded_user1, st.session_state["embedded_tastes"])[0].numpy()
            similarities2 = util.cos_sim(embedded_user2, st.session_state["embedded_tastes"])[0].numpy()
            
            # Compute preference score
            tastes_df = st.session_state["Tastes"].copy()
            tastes_df["sim1"] = similarities1
            tastes_df["sim2"] = np.where(similarities2 > 0, similarities2, - similarities2)
            tastes_df["Score"] = tastes_df["sim1"] - tastes_df["sim2"]
            tastes_df = tastes_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
            
            # Save best domain
            st.session_state["Tastes"] = tastes_df
            st.session_state["domain"] = tastes_df.loc[0, "Domain"]
            st.session_state["step"] = 2
            st.rerun()

# Domain and skills questions
elif st.session_state["step"] == 2:
    st.header(f"Your Best Match Domain")
    st.success(f"Most related domain: {st.session_state['domain']}")
    
    Skills_domain = Skills[Skills["Domain"] == st.session_state["domain"]]
    
    if len(Skills_domain) == 0:
        st.error(f"⚠️ No skills found for domain: **{st.session_state['domain']}")
        st.info("Try another domain.")
        
        available_domains = Skills["Domain"].unique()
        st.write("**Available domains with skills:**")
        st.write(", ".join(available_domains))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start Over", use_container_width=True):
                reset_app()
        with col2:
            if st.button("⬅️ Go Back", use_container_width=True):
                st.session_state["step"] = 1
                st.rerun()
    else:
        st.divider()
        st.subheader(f"Tell us about your experience in '{st.session_state['domain']}'")
        
        # Collect user info about domain
        user_input3 = st.text_area(
            f"💼 What are your competencies in '{st.session_state['domain']}' domain?",
            height=150,
            value=st.session_state["user_input3"],
            key="input3_widget"
        )
        user_input4 = st.text_area(
            f"📂 What projects have you done in '{st.session_state['domain']}' domain?",
            height=150,
            value=st.session_state["user_input4"],
            key="input4_widget"
        )
        question = Questions[Questions["Domain"] == st.session_state["domain"]]["Questions"].iloc[0]
        user_input5 = st.text_area(
            f"➕ {question}",
            height=150,
            value=st.session_state["user_input5"],
            key="input5_widget"
        )
        
        if st.button("Analyze My Profile", type="primary", disabled=(user_input3.strip() == "" or user_input4.strip() == "" or user_input5.strip() == "")):
            st.session_state["user_input3"] = user_input3
            st.session_state["user_input4"] = user_input4
            st.session_state["user_input5"] = user_input5
            
            with st.spinner("Analyzing your skills and matching jobs..."):
                # Embed and compute similarities
                st.session_state["Skills_competency"] = Skills_domain[["Competency", "Skills", "Weight"]].reset_index(drop=True)
                st.session_state["embedded_skills"] = st.session_state["model"].encode(
                    st.session_state["Skills_competency"]["Skills"].apply(normalize),
                    convert_to_tensor=True
                )
                
                cleaned_user_input3 = normalize(st.session_state["user_input3"])
                cleaned_user_input4 = normalize(st.session_state["user_input4"])
                cleaned_user_input5 = normalize(st.session_state["user_input5"])
                
                embedded_user3 = st.session_state["model"].encode(cleaned_user_input3, convert_to_tensor=True)
                embedded_user4 = st.session_state["model"].encode(cleaned_user_input4, convert_to_tensor=True)
                embedded_user5 = st.session_state["model"].encode(cleaned_user_input5, convert_to_tensor=True)
                
                similarities3 = util.cos_sim(embedded_user3, st.session_state["embedded_skills"])[0].numpy()
                similarities4 = util.cos_sim(embedded_user4, st.session_state["embedded_skills"])[0].numpy()
                similarities5 = util.cos_sim(embedded_user5, st.session_state["embedded_skills"])[0].numpy()
                
                # Weighted score per skill
                st.session_state["Skills_competency"]["sim3"] = similarities3
                st.session_state["Skills_competency"]["sim4"] = similarities4
                st.session_state["Skills_competency"]["sim5"] = similarities5
                
                total_similarity = similarities3 + similarities4 + similarities5
                weighted_scores = total_similarity * st.session_state["Skills_competency"]["Weight"]
                total_weight = st.session_state["Skills_competency"]["Weight"].sum()
                st.session_state["Skills_competency"]["Score"] = weighted_scores / total_weight
                
                # Sort by score
                st.session_state["Skills_competency"] = (
                    st.session_state["Skills_competency"]
                    .sort_values(by="Score", ascending=False)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                
                # Aggregate results by job
                Jobs_competencies = Skills_domain.groupby("Job").apply(
                    lambda x: pd.Series({
                        'Competency': ", ".join(x['Competency'].astype(str)),
                        'Weights': dict(zip(x['Competency'], x['Weight']))
                    })
                ).reset_index()
          
                Jobs_competencies["Score"] = 0.0
                
                for competency, score in zip(st.session_state["Skills_competency"]["Competency"], st.session_state["Skills_competency"]["Score"]):
                    for index, row in Jobs_competencies.iterrows():
                        job_skills = row['Competency'].split(', ')
                        
                        if competency in job_skills:
                            Jobs_competencies.at[index, 'Score'] += score
            
                st.session_state["Jobs_competencies"] = Jobs_competencies.sort_values(by="Score", ascending=False).reset_index(drop=True)
                
                st.session_state["step"] = 3
                st.rerun()
    if st.button("🔄 Start New Search", type="primary", use_container_width=True):
        reset_app()

# Final results
elif st.session_state["step"] == 3:
    st.header("📜 Your Results")
    
    # Top 3 matching jobs
    st.subheader("🏆 Top 3 Jobs Matching Your Profile")
    top3_jobs = st.session_state["Jobs_competencies"].head(3)

    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for val in top3_jobs.iterrows():
        (i, row) = val

        with cols[i]:
            st.metric(
                label=f"{medals[i]} #{i+1}",
                value=row['Job'],
                delta=f"Score: {row['Score']:.3f}"
            )
    st.divider()
    
    # Show top competencies
    st.subheader("💡 Your Top Competencies")
    top_skills = st.session_state["Skills_competency"].head(5)
    fig_skills, ax_skills = plt.subplots(figsize=(10, 5))
    colors_skills = plt.cm.Greens(top_skills["Weight"] / top_skills["Weight"].max())
    bars = ax_skills.barh(range(len(top_skills)), top_skills["Score"], color=colors_skills)
    ax_skills.set_yticks(range(len(top_skills)))
    ax_skills.set_yticklabels(top_skills["Competency"])
    ax_skills.set_xlabel("Score", fontsize=12)
    ax_skills.set_title("Top 5 Matching Competencies (colored by weight)", fontsize=14, fontweight='bold')
    ax_skills.grid(axis='x', alpha=0.3)
    ax_skills.invert_yaxis()
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=plt.Normalize(vmin=top_skills["Weight"].min(), vmax=top_skills["Weight"].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_skills)
    cbar.set_label('Weight', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig_skills)
    plt.close()
    
    # Radar chart for similarity
    st.subheader("🎯 Similarity Analysis")
    radar_data = top_skills[["Competency", "sim3", "sim4", "sim5"]].head(5)
    categories = ['Competencies', 'Projects', 'Specific Question']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(radar_data)))
    
    for idx, (i, row) in enumerate(radar_data.iterrows()):
        values = [row['sim3'], row['sim4'], row['sim5']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Competency'], color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Similarity Scores Across Different Inputs", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    with st.expander("📋 Detailed Competency Scores"):
        st.dataframe(
            st.session_state["Skills_competency"][["Competency", "Skills", "Weight", "Score"]],
            use_container_width=True
        )
    
    st.divider()
  
    if st.button("🔄 Start New Search", type="primary", use_container_width=True):
        reset_app()