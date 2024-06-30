import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd  # Import pandas for CSV file handling
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Retrieve the token from the environment variables
llama3_token = os.getenv('LLAMA3_TOKEN')

# Initialize the OpenAI client
client = OpenAI(
    api_key=llama3_token,
    base_url="https://api.llama-api.com"
)

# Streamlit app
st.title('Windy City Health Coders - Medical Transcripts GenAI with LLaMA 3')

# Add custom CSS for font sizes and input box styling
st.markdown("""
    <style>
        .big-font {
            font-size: 24px !important;
            color: #333;
        }
        .medium-font {
            font-size: 18px !important;
            color: #666;
        }
        .text-input-label {
            font-size: 24px !important;
            color: #333;
        }
        .dataframe {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .text-input {
            font-size: 18px !important;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #007bff;
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# File uploader widget with larger font
st.markdown('<p class="big-font">Upload a CSV file</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="csv")

# Initialize the user query as None
user_query = None

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None)  # Show full length of each column
    
    # Display the dataset with a larger font size
    st.markdown('<p class="big-font">Here is the uploaded dataset:</p>', unsafe_allow_html=True)
    st.dataframe(data, use_container_width=True)  # Use st.dataframe() with full container width
    
    # Add a text input box for the user query with a larger font size and placeholder text
    st.markdown('<p class="text-input-label">Ask me anything about this dataset:</p>', unsafe_allow_html=True)
    user_query = st.text_input("AAA", placeholder="Type your question and press Enter...", label_visibility="hidden", help="Enter your question here", key="user_query")

else:
    # Add a text input box for the user query with a larger font size and placeholder text
    st.markdown('<p class="text-input-label">Ask me anything about this dataset:</p>', unsafe_allow_html=True)
    user_query = st.text_input("AAA", placeholder="Type your question and press Enter...", label_visibility="hidden", help="Enter your question here", key="user_query")

# Check if the user_query is not empty
if user_query:
    if uploaded_file is not None:
        # Send request to the OpenAI API with dataset information if a file is uploaded
        response = client.chat.completions.create(
            model="llama-13b-chat",
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": f"Here's a dataset: {data.head().to_dict()}.\n\n{user_query}"}
            ],
            max_tokens=1000
        )
    else:
        # Send request to the OpenAI API without dataset information if no file is uploaded
        response = client.chat.completions.create(
            model="llama-13b-chat",
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": user_query}
            ],
            max_tokens=1000
        )

    # Get the response content
    response_answer = response.choices[0].message.content

    # Display the response
    st.write(response_answer)
