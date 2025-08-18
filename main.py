import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

url1, url2, url3 = st.sidebar.text_input("URL 1"), st.sidebar.text_input("URL 2"), st.sidebar.text_input("URL 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if not urls:
        placeholder.text("You must provide at least one valid URL.")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            unique_sources = list(set(sources))
            for source in unique_sources:
                st.write(source)

    except RuntimeError as e:
        placeholder.text("You must process urls first.")