import streamlit as st
from datetime import datetime

description = """
**Directions App** is your indispensable travel companion, offering precise navigation solutions for adventurers and everyday explorers alike. Whether you're discovering new horizons, effortlessly finding your way back to your parked car, sharing locations seamlessly, or transforming photos into routes, Directions App empowers you to navigate anywhere on Earth without the need for traditional addresses. As we celebrate our launch, we're delighted to extend a free lifetime upgrade to our Premium version for a limited time, ensuring your journeys are guided with ease. Join us in liberating your favorite places from proprietary apps, simplifying meetups, and exploring the world, one destination at a time.i"""


def sidebar():
    with st.sidebar:
        if "session_chat_history" in st.session_state:
            # st.divider()
            # Generate a unique filename using the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"chat_history_{timestamp}.txt"
            chat_history = st.session_state.session_chat_history
            chat_text = "\n".join(
                [f"Query: {query}\nAnswer: {answer}\n-----" for query, answer in chat_history])
            if not chat_history:
                st.download_button('Download Session Chat', chat_text,
                                   file_name=file_name, use_container_width=True, disabled=True)
            else:
                st.download_button('Download Session Chat', chat_text,
                                   file_name=file_name, use_container_width=True)
        st.divider()
        st.header("About")
        st.write(f"{description}")
