import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if not api_key:
    st.error("Google API Key not found. Please set it in a .env file.")
    st.stop() # Stop the Streamlit app if no API key is found

# Configure the Google Generative AI library with your API key
genai.configure(api_key=api_key)

# Initialize the Gemini model

model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# Function to generate email and summary using two sequential LLM calls
def generate_email_and_summary(bullets_input):
    """
    Generates a formal email from bullet points and then summarizes the email.

    Args:
        bullets_input (str): A string containing key bullet points entered by the user.

    Returns:
        tuple: A tuple containing the generated email (str) and its summary (str).
    """
    # Step 1: Convert bullet points to a formal email
    # The prompt instructs the model to act as a professional email writer.
    # It emphasizes formality, clarity, and conciseness.
    prompt1 = f"""You are a professional email writer.
Convert the following bullet points into a formal, clear, and concise email:

Bullet Points:
{bullets_input}

Email:"""
    try:
        # Generate content for the email
        response1 = model.generate_content(prompt1)
        email = response1.text
    except Exception as e:
        st.error(f"Error generating email: {e}")
        return None, None

    # Step 2: Summarize the generated email
    # The prompt instructs the model to summarize the email in 2-3 lines (TL;DR).
    prompt2 = f"""Summarize the following email in 2-3 lines:

Email:
{email}

Summary:"""
    try:
        # Generate content for the summary
        response2 = model.generate_content(prompt2)
        summary = response2.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return email, None # Return email even if summary fails

    return email.strip(), summary.strip()

# --- Streamlit User Interface (UI) ---
# Set basic page configuration for the Streamlit app
st.set_page_config(page_title="Bullet2Mail", page_icon="📧", layout="centered")

# Display the main title and a brief description
st.title("📧 Bullet2Mail: Email Generator from Bullet Points")
st.write("Enter bullet points below and generate a professional email with a short summary.")

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
            # Call the function to generate email and summary
            email, summary = generate_email_and_summary(bullets_input)

            # Display the results if generation was successful
            if email:
                st.subheader("📨 Generated Email:")
                st.success(email) # Using st.success for green background

                if summary:
                    st.subheader("📝 TL;DR Summary:")
                    st.info(summary) # Using st.info for blue background
                else:
                    st.warning("Could not generate summary.")
