import streamlit as st

from src.inference import (
    load_translation_model,
    translate
)

st.set_page_config(
    page_title="English Hindi Legal Translator",
    page_icon="⚖️"
)

@st.cache_resource
def get_model():

    model, english_sp, hindi_sp = load_translation_model()

    return model, english_sp, hindi_sp


model, english_sp, hindi_sp = get_model()

st.title(
    "⚖️ English-Hindi Legal Translator"
)

st.write(
    """
    Transformer-based English to Hindi
    legal translation system.
    
    Domain: Legal Documents
    """
)


text = st.text_area(
    "Enter English legal text"
)

if st.button("Translate"):

    if text.strip():

        result = translate(
            text,
            model,
            english_sp,
            hindi_sp
        )

        st.subheader(
            "Hindi Translation"
        )

        st.success(result)

    else:

        st.warning(
            "Please enter legal text."
        )

st.sidebar.title(
    "Project Details"
)

st.sidebar.write(
    """
    Model:
    Transformer Encoder-Decoder

    Framework:
    PyTorch

    Tokenizer:
    SentencePiece BPE

    Task:
    English → Hindi Legal Translation
    """
)