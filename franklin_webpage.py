import streamlit as st
import openai
import pandas as pd
import os
from dotenv import load_dotenv
from dotenv import find_dotenv
import ast  # for converting embeddings saved as strings back to arrays
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import numpy as np

# initialize OpenAI API key
load_dotenv(find_dotenv())
openai.api_key  = os.getenv('OPENAI_API_KEY')
# load DataFrame
df = pd.read_csv("franklin_embeddings.csv")

# GPT model to use
GPT_MODEL = "gpt-3.5-turbo"


def main():
    
    def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """
        Returns a list of strings and relatednesses, sorted from most related to least.
        """
        query_embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding = np.array(query_embedding_response["data"][0]["embedding"])
        strings_and_relatednesses = [
            (
                row["text"],
                relatedness_fn(query_embedding.flatten(), np.array(ast.literal_eval(row["embedding"])).flatten())
            )
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]




    def num_tokens(text: str, model: str = GPT_MODEL) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        print(len(encoding.encode(text)))
        return len(encoding.encode(text))


    def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = strings_ranked_by_relatedness(query, df)
        introduction = 'Use the below articles on Franklin webpage to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\nNext:\n"""\n{string}\n"""'
            if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_article
        print(message)        
        return message + question


    def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = query_message(query, df, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        
        messages = [
            {"role": "system", "content": f"""You answer questions about the Franklin Webpage. The format of the answer should be list of steps."""},
            {"role": "user", "content": message},
            ]
        response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0
            )
        response_message = response["choices"][0]["message"]["content"]
        return response_message

    st.title("Franklin + GPT")

    # Ask user for query
    query = st.text_input("Enter your query:")

    # Call ask() function when user presses "Submit" button
    if st.button("Submit"):
        response = ask(str(query))
        print(response)
        st.write("Response:")
        st.write(response)

    response2 = ask('I would like to add a new section on my Franklin webpage, including both an image and text?')
    st.write(response2)


if __name__ == '__main__':
    main()

