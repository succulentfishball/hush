import streamlit as st
from datetime import datetime

from .tfl_api import search_station, get_journey_options, get_line_status
from .parser import parse_journey
from .scoring import calculate_sensory_score, rank_routes, get_recommendation, get_stress_level_emoji
from .viz import create_crowding_chart, create_hourly_prediction


def main():
    st.set_page_config(
        page_title="🚇 Sensory-Safe Router",
        page_icon="🚇",
        layout="wide"
    )

    # Header
    st.title("🚇 Sensory-Safe Router")
    st.markdown("""
    **A smarter way to navigate London for neurodiverse travelers.**  
    We don't just find the fastest route—we find the *calmest* one.
    """)

    st.divider()

    # Sidebar for input
    with st.sidebar:
        st.header("🗺️ Plan Your Journey")

        # Origin
        origin_query = st.text_input("From (Station Name)", placeholder="e.g., Kings Cross")
        origin_options = []
        origin_id = None

        if origin_query and len(origin_query) >= 3:
            origin_options = search_station(origin_query)
            if origin_options:
                origin_names = [s.get("name", "") for s in origin_options]
                selected_origin = st.selectbox("Select origin station:", origin_names)
                origin_id = selected_origin  # Use station name directly
                origin_name = selected_origin

        # Destination
        dest_query = st.text_input("To (Station Name)", placeholder="e.g., Waterloo")
        dest_options = []
        dest_id = None
        dest_name = None

        if dest_query and len(dest_query) >= 3:
            dest_options = search_station(dest_query)
            if dest_options:
                dest_names = [s.get("name", "") for s in dest_options]
                selected_dest = st.selectbox("Select destination station:", dest_names)
                dest_id = selected_dest  # Use station name directly
                dest_name = selected_dest

        # Time selection
        st.subheader("⏰ When?")
        travel_option = st.radio("Departure time:", ["Now", "Custom time"])

        if travel_option == "Now":
            travel_time = datetime.now()
        else:
            col1, col2 = st.columns(2)
            with col1:
                travel_date = st.date_input("Date", datetime.now())
            with col2:
                travel_hour = st.time_input("Time", datetime.now())
            travel_time = datetime.combine(travel_date, travel_hour)

        st.divider()

        # Search button
        search_clicked = st.button("🔍 Find Calm Routes", type="primary", use_container_width=True)

    # Main content area
    if search_clicked and origin_id and dest_id:
        with st.spinner("🔄 Analyzing routes and crowding levels..."):
            # Fetch data
            journeys = get_journey_options(origin_id, dest_id, travel_time)
            line_status = get_line_status()

            if not journeys:
                st.error("❌ No routes found. Please check your stations and try again.")
                return

            # Parse and score routes
            routes = []
            for journey in journeys:
                route = parse_journey(journey, travel_time)
                route = calculate_sensory_score(route, line_status)
                routes.append(route)

            # Rank by sensory score
            ranked_routes = rank_routes(routes)

        # Display results
        st.header("🎯 Route Options (Ranked by Calmness)")

        # Recommendation banner
        recommendation = get_recommendation(ranked_routes)
        if recommendation:
            st.info(recommendation)

        # Display each route
        for i, route in enumerate(ranked_routes):
            stress_emoji = get_stress_level_emoji(route.sensory_score)
            is_recommended = i == 0

            with st.expander(
                f"{'⭐ RECOMMENDED: ' if is_recommended else ''}"
                f"Route {i+1} — {route.total_duration} mins | {stress_emoji} (Score: {route.sensory_score})",
                expanded=is_recommended
            ):
                # Route summary columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("⏱️ Total Time", f"{route.total_duration} mins")

                with col2:
                    st.metric("🧠 Sensory Score", f"{route.sensory_score}")

                with col3:
                    st.metric("🔄 Changes", f"{len(route.legs) - 1}")

                # Delay warning
                if route.has_delays:
                    st.warning(f"⚠️ **Delays detected:** {route.delay_info}")

                # Journey steps
                st.subheader("📍 Journey Steps")
                for j, leg in enumerate(route.legs):
                    mode_emoji = {
                        "tube": "🚇",
                        "bus": "🚌",
                        "walking": "🚶",
                        "dlr": "🚈",
                        "overground": "🚆",
                        "elizabeth-line": "🟣"
                    }.get(leg.mode.lower(), "🚃")

                    st.markdown(f"""
                    **{j+1}. {mode_emoji} {leg.line_name}** ({leg.duration_minutes} mins)  
                    {leg.departure_point} → {leg.arrival_point}  
                    *{leg.instruction}*
                    """)

                # Crowding visualization
                chart = create_crowding_chart(route)
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"crowding_chart_route_{i}")

        # Hourly prediction for destination
        st.divider()
        st.header("📈 Crowding Forecast")
        st.markdown("See predicted crowding at your destination over the next few hours.")

        if dest_id:
            prediction_chart = create_hourly_prediction(dest_id)
            st.plotly_chart(prediction_chart, use_container_width=True, key="hourly_prediction_chart")

    elif search_clicked:
        st.warning("⚠️ Please select both origin and destination stations.")

    else:
        # Welcome / instruction state
        st.markdown("""
        ### 👋 Welcome!

        This app helps you find the **calmest route** through London's transport network.

        **How it works:**
        1. Enter your start and end stations
        2. We fetch multiple route options from TfL
        3. For each route, we analyze:
           - 📊 **Real-time crowding** at every station
           - ⚠️ **Service disruptions** and delays
           - 🔄 **Number of interchanges** (stressful!)
        4. We calculate a **Sensory Score** and rank routes

        **Why does this matter?**

        > *"For 20% of Londoners, the fastest route is the wrong route.  
        > Anxiety and sensory overload make the Tube inaccessible."*

        ---

        👈 **Enter your journey details in the sidebar to get started!**
        """)

        # Show live line status
        st.header("🚦 Current Line Status")
        with st.spinner("Fetching live status..."):
            status = get_line_status()

            if status:
                cols = st.columns(4)
                for i, (line_id, info) in enumerate(status.items()):
                    col = cols[i % 4]
                    with col:
                        if info["has_severe_delays"]:
                            st.error(f"**{line_id.title()}**\n{info['status']}")
                        else:
                            st.success(f"**{line_id.title()}**\n{info['status']}")
