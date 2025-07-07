import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import streamlit as st


load_dotenv()


if "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
else:

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
  
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True 
)


translator = Agent(
    name="Translator Agent",
    instructions="""You are a professional translator agent.
    Your task is to accurately and naturally translate text from a given source language to a target language.
    Maintain the original meaning, tone, and context as much as possible.
    If the source or target language is not explicitly mentioned or is unclear, assume common languages like English or the most probable language based on the input.
    Provide only the translated text as your response, without any additional commentary or formatting unless specifically requested.
    """,
)


async def generate_translation(text_to_translate, source_lang, target_lang):
   
    user_prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n'{text_to_translate}'"
    
   
    return await Runner.run(
        translator,
        input=user_prompt,
        run_config=config
    )


st.set_page_config(layout="centered", page_title="AI Translator Agent")
st.title("🌐 AI Translator Agent")
st.markdown("Enter the text you want to translate, specify the source and target languages, and let the AI do the rest!")


text_input = st.text_area("Enter text to translate:", height=150, placeholder="Type your text here...")

# Dropdowns for source and target languages

languages = {
    "Auto-detect": "auto",
    "English": "English",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Chinese (Simplified)": "Chinese (Simplified)",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Arabic": "Arabic",
    "Hindi": "Hindi",
    "Urdu": "Urdu",
    "Russian": "Russian",
}

col1, col2 = st.columns(2)
with col1:
    source_lang_display = st.selectbox("Source Language:", list(languages.keys()), index=0) # Default to Auto-detect
    source_lang = languages[source_lang_display]
with col2:
    target_lang_display = st.selectbox("Target Language:", list(languages.keys()), index=1) # Default to English
    target_lang = languages[target_lang_display] 

# Button to trigger translation
if st.button("Translate", use_container_width=True):
    if text_input.strip() == "":
        st.warning("Please enter some text to translate.")
    elif source_lang == target_lang and source_lang != "auto":
        st.warning("Source and target languages cannot be the same (unless source is 'Auto-detect').")
    else:
        with st.spinner("Translating..."):
            # Run the async function
            translation_response = asyncio.run(generate_translation(text_input, source_lang, target_lang))
            
            # Display output
            if translation_response and translation_response.final_output:
                st.success("Translation complete:")
                st.write(translation_response.final_output)
            else:
                st.error("Could not get a translation. Please try again.")

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .stTextArea textarea {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
