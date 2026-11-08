import streamlit as st

def show_bookings():
    st.divider()
    st.subheader("🟨 Bookings & 🟥 1st Half Red")
    pred_cards = 4.2
    red_1st_half = 12
    c1, c2 = st.columns(2)
    c1.metric("Total Cards", f"{pred_cards:.1f}")
    c2.metric("Red in 1st Half", f"{red_1st_half}%")
    if pred_cards >= 4.0:
        st.success("✅ OVER 3.5 Cards")
    else:
        st.info("UNDER 3.5 Cards")
    if red_1st_half >= 20:
        st.error("🔥 HIGH - Red in 1st Half possible")
    else:
        st.success("✅ LOW - No Red in 1st Half likely")
