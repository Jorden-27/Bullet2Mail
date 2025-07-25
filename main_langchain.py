import os
import streamlit as st
from dotenv import load_dotenv

# Import necessary LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the API key from .env file
# Ensure you have a .env file in the same directory as this script
# with a line like: GOOGLE_API_KEY="YOUR_API_KEY_HERE"
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if not api_key:
    st.error("Google API Key not found. Please set it in a .env file.")
    st.stop() # Stop the Streamlit app if no API key is found

# Initialize the LangChain Google Generative AI model
# Using 'gemini-2.5-pro' as per your Google AI Pro plan access.
# LangChain handles the underlying API calls to Google Generative AI.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)

# --- Define LangChain components for Email Generation ---

# 1. Prompt Template for Email Generation
# This template defines how the bullet points will be formatted for the LLM.
email_prompt_template = PromptTemplate(
    input_variables=["bullets"],
    template="""You are a professional email writer.
Convert the following bullet points into a formal, clear, and concise email:

Bullet Points:
{bullets}

Email:"""
)

# 2. LLMChain for Email Generation
# This chain combines the LLM and the email prompt template.
email_chain = LLMChain(llm=llm, prompt=email_prompt_template, verbose=False) # Set verbose=True for debugging prompt/response

# --- Define LangChain components for Summary Generation ---

# 1. Prompt Template for Summary Generation
# This template defines how the email will be formatted for summarization.
summary_prompt_template = PromptTemplate(
    input_variables=["email"],
    template="""Summarize the following email in 2-3 lines:

Email:
{email}

Summary:"""
)

# 2. LLMChain for Summary Generation
# This chain combines the LLM and the summary prompt template.
summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template, verbose=False) # Set verbose=True for debugging prompt/response


# Function to generate email and summary using LangChain LLMChains
def generate_email_and_summary_langchain(bullets_input):
    """
    Generates a formal email from bullet points and then summarizes the email using LangChain.

    Args:
        bullets_input (str): A string containing key bullet points entered by the user.

    Returns:
        tuple: A tuple containing the generated email (str) and its summary (str).
    """
    email = None
    summary = None

    try:
        # Step 1: Generate the email using the email_chain
        # The .invoke() method passes the input variables to the chain.
        email_response = email_chain.invoke({"bullets": bullets_input})
        email = email_response['text'] # LangChain's invoke returns a dictionary with 'text' key
    except Exception as e:
        st.error(f"Error generating email: {e}")
        return None, None # If email generation fails, stop here

    try:
        # Step 2: Summarize the generated email using the summary_chain
        # The generated email from Step 1 is passed as input to the summary chain.
        summary_response = summary_chain.invoke({"email": email})
        summary = summary_response['text'] # LangChain's invoke returns a dictionary with 'text' key
    except Exception as e:
        st.warning(f"Error generating summary: {e}")
        # Return email even if summary generation fails, as the email might still be useful
        return email.strip(), None

    return email.strip(), summary.strip()

# --- Streamlit User Interface (UI) ---
# Set basic page configuration for the Streamlit app
st.set_page_config(page_title="Bullet2Mail (LangChain)", page_icon="📧", layout="centered")

# Display the main title and a brief description
st.title("📧 Bullet2Mail: Email Generator from Bullet Points (LangChain)")
st.write("Enter bullet points below and generate a professional email with a short summary using LangChain.")

# Text area for user to input bullet points
# Provides a placeholder for guidance
bullets_input = st.text_area(
    "✏️ Enter your bullet points:",
    height=200,
    placeholder="- Mention project update\n- Request meeting for discussion\n- Propose next steps"
)

# Button to trigger the email generation process
if st.button("Generate Email"):
    # Check if the input text area is empty
    if not bullets_input.strip():
        st.warning("Please enter bullet points to proceed.")
    else:
        # Display a spinner while generation is in progress
        with st.spinner("Generating email and summary..."):
            # Call the LangChain-based function
            email, summary = generate_email_and_summary_langchain(bullets_input)

            # Display the results if generation was successful
            if email:
                st.subheader("📨 Generated Email:")
                st.success(email) # Using st.success for green background

                if summary:
                    st.subheader("📝 TL;DR Summary:")
                    st.info(summary) # Using st.info for blue background
                else:
                    st.warning("Could not generate summary.")
