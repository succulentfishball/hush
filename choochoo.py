"""Thin entrypoint for Sensory-Safe Router (Streamlit app).

This file now only delegates to the modular package `sensory_router`.
Run: streamlit run choochoo.py
"""

from sensory_router.ui import main

if __name__ == "__main__":
    main()

    