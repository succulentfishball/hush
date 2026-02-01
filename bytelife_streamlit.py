"""
ByteLife — The Data Simulator
Streamlit Version

A life simulator that uses predictive models (income quantile regression & life satisfaction
classification) trained from real social science data. Players make narrative choices that 
shape their character vector, then receive predictions for income and satisfaction.

Usage:
    streamlit run bytelife_streamlit.py
"""

import streamlit as st
from typing import Callable
import os

from bytelife_models import (
    ByteLifeVector,
    ByteLifeModels,
    get_choice_outcome,
    DEFAULT_PREDICTION,
)


INITIAL_VECTOR = ByteLifeVector(
    age_dv=25,
    nchild_dv=0,
    hiqual_dv=3,
    hhsize=1,
    ever_married=0,
    sex_female=0,
    occ_2=0,
    occ_3=0,
)

# Initialize models at startup
ML_MODELS = ByteLifeModels()


def init_session_state():
    """Initialize Streamlit session state for the game."""
    if "vector" not in st.session_state:
        st.session_state.vector = INITIAL_VECTOR
    if "gameState" not in st.session_state:
        st.session_state.gameState = "start"
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "narrativeText" not in st.session_state:
        st.session_state.narrativeText = ""
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "isBusy" not in st.session_state:
        st.session_state.isBusy = False


def start_life():
    """Randomize initial conditions and begin the game."""
    import random
    
    female = 1 if random.random() > 0.5 else 0
    hhRoll = random.random()
    
    if hhRoll < 0.70:
        initialHh = 3
        genesisStory = "You were born into a warm home with both parents present."
    elif hhRoll < 0.95:
        initialHh = 2
        genesisStory = "You were raised in a resilient single-parent household."
    else:
        initialHh = 1
        genesisStory = "Life started tough; you grew up in the system with no immediate family."
    
    vector = INITIAL_VECTOR
    vector.sex_female = female
    vector.hhsize = initialHh
    
    st.session_state.vector = vector
    st.session_state.narrativeText = genesisStory
    st.session_state.gameState = "narrative"
    st.session_state.step = 1


def handle_choice(label: str, update_fn: Callable, nextStep: int):
    """Process a player's choice and advance the game state."""
    st.session_state.isBusy = True
    
    # Update vector
    st.session_state.vector = update_fn(st.session_state.vector)
    
    # Get narrative outcome
    outcome = get_choice_outcome(f"Scenario {st.session_state.step}", label)
    st.session_state.narrativeText = outcome
    st.session_state.gameState = "narrative"
    st.session_state.step = nextStep
    st.session_state.isBusy = False


def render_scenario():
    """Render the current scenario based on step number."""
    step = st.session_state.step
    vector = st.session_state.vector
    
    if step == 1:
        st.subheader("🎓 Your parents ask about your future. A-Levels or find work?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Take A-Levels", use_container_width=True, key="btn_1a"):
                handle_choice("Take A-Levels", lambda v: ByteLifeVector(**{**v.to_dict(), "hiqual_dv": 2}), 2)
        with col2:
            if st.button("Find Work Now", use_container_width=True, key="btn_1b"):
                handle_choice("Find Work", lambda v: ByteLifeVector(**{**v.to_dict(), "hiqual_dv": 3}), 4)
    
    elif step == 2:
        st.subheader("🎓 University applications are open. Do you apply?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply to University", use_container_width=True, key="btn_2a"):
                handle_choice("Apply", lambda v: ByteLifeVector(**{**v.to_dict(), "hiqual_dv": 1}), 3)
        with col2:
            if st.button("Just get a diploma", use_container_width=True, key="btn_2b"):
                handle_choice("Skip", lambda v: ByteLifeVector(**{**v.to_dict(), "hiqual_dv": 2}), 4)
    
    elif step == 3:
        st.subheader("🎯 What degree do you choose?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Computer Science", use_container_width=True, key="btn_3a"):
                handle_choice("Computer Science", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 0, "occ_3": 0}), 5)
        with col2:
            if st.button("Business Admin", use_container_width=True, key="btn_3b"):
                handle_choice("Business Admin", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 1, "occ_3": 0}), 5)
        with col3:
            if st.button("Fine Arts", use_container_width=True, key="btn_3c"):
                handle_choice("Fine Arts", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 0, "occ_3": 1}), 5)
    
    elif step == 4:
        st.subheader("💼 Which sector do you apply for?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Office Admin", use_container_width=True, key="btn_4a"):
                handle_choice("Office Admin", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 1, "occ_3": 0}), 5)
        with col2:
            if st.button("Retail Sales", use_container_width=True, key="btn_4b"):
                handle_choice("Retail", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 0, "occ_3": 1}), 5)
        with col3:
            if st.button("Plumbing Apprentice", use_container_width=True, key="btn_4c"):
                handle_choice("Apprenticeship", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "occ_2": 0, "occ_3": 1}), 5)
    
    elif step == 5:
        st.subheader("💍 \"Will you marry me?\" your partner asks.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, forever!", use_container_width=True, key="btn_5a"):
                handle_choice("Yes", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "ever_married": 1, "hhsize": v.hhsize + 1}), 6)
        with col2:
            if st.button("I'm not ready.", use_container_width=True, key="btn_5b"):
                handle_choice("No", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "ever_married": 0}), 6)
    
    elif step == 6:
        st.subheader("🏠 Thinking of moving out?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Pack your bags", use_container_width=True, key="btn_6a"):
                handle_choice("Move Out", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "hhsize": max(1, v.hhsize - 2)}), 7)
        with col2:
            if st.button("Stay with parents", use_container_width=True, key="btn_6b"):
                handle_choice("Stay", lambda v: v, 8)
    
    elif step == 7:
        st.subheader("👶 \"What do you think about having kids?\"")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Just one", use_container_width=True, key="btn_7a"):
                handle_choice("One child", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "nchild_dv": 1, "hhsize": v.hhsize + 1}), 9)
        with col2:
            if st.button("A full house (2)", use_container_width=True, key="btn_7b"):
                handle_choice("Two kids", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "nchild_dv": 2, "hhsize": v.hhsize + 2}), 9)
        with col3:
            if st.button("Not for me", use_container_width=True, key="btn_7c"):
                handle_choice("None", lambda v: v, 9)
    
    elif step == 8:
        st.subheader("👨‍👩‍👧 Your parents ask if they can stay in your spare room.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Welcome them", use_container_width=True, key="btn_8a"):
                handle_choice("Yes", 
                            lambda v: ByteLifeVector(**{**v.to_dict(), "hhsize": v.hhsize + 2}), 9)
        with col2:
            if st.button("Help them find a flat", use_container_width=True, key="btn_8b"):
                handle_choice("No", lambda v: v, 9)
    
    elif step == 9:
        st.session_state.gameState = "age_select"
        st.rerun()


def render_results():
    """Render the final predictions and life summary."""
    pred = st.session_state.prediction
    vector = st.session_state.vector
    
    st.success(f"✨ {pred['summary']}")
    
    # Income Quantiles
    st.subheader("💰 Predicted Annual Income Quantiles")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("P10 (Bottom 10%)", f"${pred['income']['p10']:,}")
    with col2:
        st.metric("P25", f"${pred['income']['p25']:,}")
    with col3:
        st.metric("P50 (Median)", f"${pred['income']['p50']:,}")
    with col4:
        st.metric("P75", f"${pred['income']['p75']:,}")
    with col5:
        st.metric("P90 (Top 10%)", f"${pred['income']['p90']:,}")
    
    # Life Satisfaction
    st.subheader("😊 Life Satisfaction Probability Distribution")
    sat_data = {
        "Scale": list(range(1, 8)),
        "Probability": [p * 100 for p in pred['satisfaction']['probabilities']],
    }
    
    import pandas as pd
    df_sat = pd.DataFrame(sat_data)
    st.bar_chart(df_sat.set_index("Scale"), use_container_width=True)
    
    st.write(f"**Most likely satisfaction level**: {pred['satisfaction']['mostLikely']} (on 1-7 scale)")
    
    # Restart button
    if st.button("🔄 Restart Life", use_container_width=True):
        st.session_state.gameState = "start"
        st.session_state.vector = INITIAL_VECTOR
        st.session_state.step = 0
        st.session_state.prediction = None
        st.rerun()


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Page config
    st.set_page_config(
        page_title="ByteLife — The Data Simulator",
        page_icon="🧬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        body {
            background-color: #09090b;
            color: #fafafa;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 600px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("# 🧬 BYTELIFE")
        st.caption("Statistical Genesis — Your life, by the numbers")
    with col2:
        st.markdown("❤️")
    
    st.divider()
    
    # Game states
    if st.session_state.gameState == "start":
        st.markdown("""
        ## Ready to be born?
        
        Your gender, family, and initial wealth will be **randomly assigned** from global statistics.
        
        Then navigate **9 major life choices** that shape your character:
        - Education & Career
        - Marriage & Family
        - Housing & Living Arrangements
        
        Finally, **machine learning models** will predict your life outcomes:
        - Income distribution (P10, P25, P50, P75, P90)
        - Life satisfaction probability (1–7 scale)
        """)
        
        if st.button("🎮 Start Your Life", use_container_width=True, type="primary"):
            start_life()
            st.rerun()
    
    elif st.session_state.gameState == "narrative":
        # Display narrative
        st.info(f"📖 {st.session_state.narrativeText}")
        
        if st.button("Continue →", use_container_width=True):
            st.session_state.gameState = "playing"
            st.rerun()
    
    elif st.session_state.gameState == "playing":
        # Display character status
        vector = st.session_state.vector
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gender", "♀️ Female" if vector.sex_female else "♂️ Male")
        with col2:
            st.metric("Family", f"{vector.hhsize} people")
        with col3:
            st.metric("Qual Level", 6 - vector.hiqual_dv)
        with col4:
            st.metric("Married", "💍 Yes" if vector.ever_married else "No")
        
        st.divider()
        
        # Render current scenario
        render_scenario()
    
    elif st.session_state.gameState == "age_select":
        st.subheader("⏸️ Freeze Time")
        st.write("Enter the age at which you want to finalize your life's statistical results.")
        
        age = st.number_input("Age", min_value=25, max_value=65, value=st.session_state.vector.age_dv, step=1)
        st.session_state.vector.age_dv = int(age)
        
        if st.button("🔮 Finalize Simulation", use_container_width=True, type="primary"):
            with st.spinner("Running Quantile Models..."):
                pred = ML_MODELS.predict(st.session_state.vector)
                if pred is None:
                    pred = DEFAULT_PREDICTION
                st.session_state.prediction = pred
                st.session_state.gameState = "results"
            st.rerun()
    
    elif st.session_state.gameState == "results":
        render_results()


if __name__ == "__main__":
    main()
