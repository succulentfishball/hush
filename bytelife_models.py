"""
ByteLife Models Module
======================
Handles model loading, prediction logic, and data structures.
"""

import pickle
import json
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class ByteLifeVector:
    """Character state vector for the simulation."""
    age_dv: int
    nchild_dv: int
    hiqual_dv: int
    hhsize: int
    ever_married: int
    sex_female: int
    occ_2: int
    occ_3: int

    def to_dict(self) -> dict:
        return {
            "age_dv": self.age_dv,
            "nchild_dv": self.nchild_dv,
            "hiqual_dv": self.hiqual_dv,
            "hhsize": self.hhsize,
            "ever_married": self.ever_married,
            "sex_female": self.sex_female,
            "occ_2": self.occ_2,
            "occ_3": self.occ_3,
        }


class ByteLifeModels:
    """Manages loading and using trained ML models."""
    
    def __init__(self, income_model_path: str = "model_income.pkl", 
                 satisfaction_model_path: str = "model_satisfaction.pkl"):
        self.income_model_path = income_model_path
        self.satisfaction_model_path = satisfaction_model_path
        self.income_models = None
        self.satisfaction_model = None
        self.models_loaded = False
        self.feature_names = ["age_dv", "nchild_dv", "hiqual_dv", "hhsize", 
                              "ever_married", "sex_female", "occ_2.0", "occ_3.0"]
        
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load pickled models from disk."""
        try:
            with open(self.income_model_path, "rb") as f:
                self.income_models = pickle.load(f)
            with open(self.satisfaction_model_path, "rb") as f:
                self.satisfaction_model = pickle.load(f)
            self.models_loaded = True
            return True
        except FileNotFoundError as e:
            print(f"⚠️  Warning: Models not found - {e}")
            return False
    
    def predict(self, vector: ByteLifeVector) -> Optional[Dict]:
        """
        Predict income quantiles and life satisfaction for a character vector.
        
        Returns:
            Dict with 'income' (p10, p25, p50, p75, p90) and 'satisfaction' 
            (probabilities, mostLikely) and 'summary'
        """
        if not self.models_loaded:
            return None
        
        # Create DataFrame with correct feature names
        features_df = pd.DataFrame([[
            vector.age_dv,
            vector.nchild_dv,
            vector.hiqual_dv,
            vector.hhsize,
            vector.ever_married,
            vector.sex_female,
            vector.occ_2,
            vector.occ_3,
        ]], columns=self.feature_names)
        
        # Income predictions (5 models for 5 quantiles)
        income = {
            "p10": max(0, int(self.income_models["P10"].predict(features_df)[0])),
            "p25": max(0, int(self.income_models["P25"].predict(features_df)[0])),
            "p50": max(0, int(self.income_models["P50"].predict(features_df)[0])),
            "p75": max(0, int(self.income_models["P75"].predict(features_df)[0])),
            "p90": max(0, int(self.income_models["P90"].predict(features_df)[0])),
        }
        
        # Life satisfaction (probability vector over 7 classes)
        sat_proba = self.satisfaction_model.predict_proba(features_df)[0]
        satisfaction = {
            "probabilities": sat_proba.tolist(),
            "mostLikely": int(np.argmax(sat_proba)) + 1,
        }
        
        summary = f"A life shaped by choices. At {vector.age_dv}, " \
                  f"with {vector.nchild_dv} children and a household of {vector.hhsize}, " \
                  f"you navigate the complexities of modern life."
        
        return {
            "income": income,
            "satisfaction": satisfaction,
            "summary": summary,
        }


# Narrative responses (pre-written to avoid API quota issues)
NARRATIVE_RESPONSES = {
    "Take A-Levels": "Your teachers see potential. You commit to two years of intense study.",
    "Find Work": "You land an entry-level job and start saving money immediately.",
    "Apply": "Your application is accepted! You begin preparing for university life.",
    "Skip": "You settle into your career path, gaining valuable work experience.",
    "Computer Science": "Your coding skills become your greatest asset in the job market.",
    "Business Admin": "Your organizational talents lead to quick promotions.",
    "Fine Arts": "You follow your passion, building a creative portfolio.",
    "Office Admin": "You become known as the most reliable person in the office.",
    "Retail": "Customer service teaches you valuable people skills.",
    "Apprenticeship": "You gain hands-on trade expertise that pays dividends later.",
    "Yes": "Love and partnership bring new dimensions to your life.",
    "No": "You focus on personal growth and career advancement.",
    "Move Out": "Independence feels exhilarating as you settle into your own space.",
    "Stay": "Family bonds deepen as you remain under the same roof.",
    "One child": "Parenthood transforms your priorities and brings joy.",
    "Two kids": "A full house means chaos, laughter, and unconditional love.",
    "None": "You dedicate yourself fully to other pursuits and ambitions.",
    "Welcome them": "Multi-generational living enriches your household.",
    "Help them find a flat": "You support their independence while building your own.",
}


def get_choice_outcome(scenario: str, choice: str) -> str:
    """
    Generate a narrative outcome for the player's choice.
    Uses pre-written responses to avoid API quota issues.
    Falls back to generic text if exact choice not found.
    """
    if choice in NARRATIVE_RESPONSES:
        return NARRATIVE_RESPONSES[choice]
    
    import random
    generic_responses = [
        "Your choice sets a new trajectory for your life.",
        "This decision opens unexpected doors for you.",
        "You move forward with newfound determination.",
        "Your path takes an interesting turn.",
        "This chapter of your life begins with hope.",
    ]
    return random.choice(generic_responses)


# Default fallback predictions (for when models are unavailable)
DEFAULT_PREDICTION = {
    "income": {"p10": 15000, "p25": 25000, "p50": 35000, "p75": 50000, "p90": 75000},
    "satisfaction": {
        "probabilities": [0.05, 0.10, 0.15, 0.25, 0.30, 0.15, 0.0],
        "mostLikely": 5,
    },
    "summary": "A simulated life, shaped by your choices.",
}
