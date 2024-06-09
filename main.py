import streamlit as st
import pandas as pd
import openai

# Set your OpenAI API key


# Function to read CSV file
def read_csv(file):
    df = pd.read_csv(file)
    return df


# Function to train using OpenAI's GPT
def train_with_openai(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text,
        max_tokens=100
    )
    return response.choices[0].text.strip()


# Streamlit App
def main():
    st.title("CSV Reader and GPT Trainer")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = read_csv(uploaded_file)
        st.dataframe(df)

        # Select a column for training
        column_name = st.selectbox("Select a column for training", df.columns)

        # Get text data from the selected column
        text_data = " ".join(df[column_name].astype(str).tolist())

        # Train with OpenAI
        st.subheader("Training with OpenAI GPT-3")
        result = train_with_openai(text_data)
        st.write(result)

        user_input = st.text_input("Enter your query::")

        if user_input is not None and user_input != "":
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Input: {user_input}\n",
                temperature=0.7,
                max_tokens=150,
            )
            st.write(response.choices[0].text.strip())

if __name__ == "__main__":
    main()
