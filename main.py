import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

# ---------------------------
# Initialize session state
# ---------------------------
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False  # Have they processed URLs yet?

# ---------------------------
# Collect URLs
# ---------------------------
url1, url2, url3 = st.sidebar.text_input("URL 1"), st.sidebar.text_input("URL 2"), st.sidebar.text_input("URL 3")

placeholder = st.empty()

urls = [url for url in (url1, url2, url3) if url != '']

# ---------------------------
# Process URLs button
# ---------------------------
if st.sidebar.button("Process URLs"):
    if not urls:
        placeholder.text("You must provide at least one valid URL.")
    else:
        # Reset vector database state before processing
        st.session_state.vector_ready = False
        for status in process_urls(urls):
            placeholder.text(status)
        st.session_state.vector_ready = True  # Now queries are allowed

# ---------------------------
# Query section
# ---------------------------
query = placeholder.text_input("Question")
if query:
    if not st.session_state.vector_ready:
        placeholder.text("You must process URLs first before asking questions.")
    else:
        try:
            answer, sources = generate_answer(query)
            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in set(sources):
                    st.write(source)

        except RuntimeError:
            placeholder.text("You must process URLs first.")