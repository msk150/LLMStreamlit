import streamlit as st
import string
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
import os


def get_completion(client, prompt, messages, model="gpt-3.5-turbo"):
    # append the new message after the previous messages
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def query_pinecone_vector_store(vectorstore, query, top_k=5, namespace='chunk_size_1000'):
    trans_preprocessing = str.maketrans("\n", " ", string.punctuation)
    query.translate(trans_preprocessing)
    docs = vectorstore.similarity_search(query, k=top_k, namespace=namespace)
    docs = [doc.page_content for doc in docs]
    return docs

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = client

    def set_prompt(self, query):
        # TODO: Set the prompt for the Obnoxious_Agent
        return f"Please tell me if the following input is obnoxious. Reply only using 'Yes' or 'No'. Here's the input: {query}"

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        return response

    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        prompt = self.set_prompt(query)
        response = get_completion(self.client, prompt, [])
        return self.extract_action(response)

    # Test function only. Not used in the final deliverables.
    def check_query_moderations(self, query):
        response = self.client.moderations.create(input=query)
        return "Yes" if response.results[0].flagged else "No"

class Query_Agent:
    def __init__(self, pinecone_index, client, embeddings) -> None:
        # TODO: Initialize the Query_Agent agent
        self.client = client
        self.vectorstore = Pinecone(index_name=pinecone_index, embedding=embeddings)

    def query_vector_store(self, query, k=5):
        # TODO: Query the Pinecone vector store
        response = query_pinecone_vector_store(self.vectorstore, query, top_k=k)
        return self.extract_action(response, query)

    def set_prompt(self, response, query):
        # TODO: Set the prompt for the Query_Agent agent
        return f"Summarize the information of the following documents: {response}. Specifically, highlight the information that is relevant to the query: {query}"

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        prompt = self.set_prompt(response, query)
        return get_completion(self.client, prompt, [])

class Answering_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the Answering_Agent
        self.client = client

    def generate_response(self, query, docs, conv_history, sub_mode=None):
        # TODO: Generate a response to the user's query
        prompt = f"Here's the query: {query}. Here's the additional information about the query. Generate a prompt for a language model to answer the query. Guide the language model to take advantage of the additional information when answering the query."
        generated_prompt = get_completion(self.client, prompt, [])

        messages = list(conv_history)
        if sub_mode is not None:
            messages.append({"role": "system", "content": f"You are an assistant who is {sub_mode}"})
        return get_completion(self.client, generated_prompt, messages)

class Relevant_Query_Agent:
    def __init__(self, client, topic) -> None:
        # TODO: Initialize the Relevant_Query_Agent
        self.client = client
        self.topic = topic

    def get_relevance(self, prompt, conv="") -> str:
        # TODO: Get if the query are relevant
        prompt = f"Here's the prompt: {prompt} and the previous conversation: {conv}. If it's a general greeting, respond 'Greeting'. If not, check if the prompt is relevant to {self.topic} based on the conversation. Reply 'Relevant' or 'Irrelevant' only. DO NOT explain the answer. Respond single word only."
        return get_completion(self.client, prompt, [{"role": "system", "content": "You are an assistant who checks if the input query is a general greeting. Then, you check if it is relevant to the topic based on the conversation."}])

class Relevant_Documents_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        self.client = client

    def get_relevance(self, prompt, docs="") -> str:
        # TODO: Get if the returned documents are relevant
        prompt = f"Here's the prompt: {prompt}.\n Here's the documents: {docs}.\n Do the documents answer the question in the prompt? Reply 'Relevant' or 'Irrelevant' only. DO NOT explain the answer. Respond single word only."
        return get_completion(self.client, prompt, [])

class Head_Agent:
    def __init__(self, pinecone_index_name='mini-project-2', topic="machine learning") -> None:
        # TODO: Initialize the Head_Agent
        # self.client = OpenAI(api_key='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')
        self.client = OpenAI()
        self.embeddings = OpenAIEmbeddings()
        self.pinecone_index_name = pinecone_index_name
        self.topic = topic

        #initialize sessionstate
        st.title("Mini Project 3: Streamlit Chatbot")
        if "openai_model" not in st.session_state:
          st.session_state.openai_model = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
          st.session_state.messages = []

        st.sidebar.title("Model Sub Mode")
        self.sub_mode = st.sidebar.selectbox("Choose the sub mode", ("default", "chatty and talkative", "concise and short"), index=0)

        # TODO: Setup the sub-agents
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.relevant_query_agent = Relevant_Query_Agent(self.client, self.topic)
        self.relevant_documents_agent = Relevant_Documents_Agent(self.client)
        self.query_agent = Query_Agent(self.pinecone_index_name, self.client, self.embeddings)
        self.answering_agent = Answering_Agent(self.client)

    def get_conversation(self):
        conversation = []
        for entry in st.session_state.messages:
            conversation.append(f"{entry['role']}: {entry['content']}\n")
        return "".join(conversation)

    #function to choose which agent
    def which_agent(self, prompt):
        if self.obnoxious_agent.check_query(prompt) == 'Yes':
            return "Please do not ask obnoxious questions."

        relevance_category = self.relevant_query_agent.get_relevance(prompt)
        if relevance_category == "Greeting":
            return get_completion(self.client, prompt, [])
        elif relevance_category == "Irrelevant":
            return f"Please ask relevant questions to the topic of {self.topic}."

        docs = self.query_agent.query_vector_store(prompt)

        relevance_category = self.relevant_documents_agent.get_relevance(prompt)
        if relevance_category == "Irrelevant":
            return f"No relevant documents found. Please ask relevant questions to the topic in the book of {self.topic}."

        sub_mode = None if self.sub_mode == "default" else self.sub_mode
        return self.answering_agent.generate_response(prompt, docs, st.session_state.messages, sub_mode)

    def main_loop(self):
        # Display existing chat messages
        # ... (code for displaying messages)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Wait for user input
        if prompt := st.chat_input("What would you like to chat about?"):
            # ... (display user message)
            with st.chat_message("user"):
                st.markdown(prompt)

            # ... (append user message to messages)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate AI response
            with st.chat_message("assistant"):
                # ... (send request to OpenAI API)
                response = self.which_agent(prompt)

                # ... (get AI response and display it)
                st.markdown(response)

            # ... (append AI response to messages)
            st.session_state.messages.append({"role": "assistant", "content": response})

# TODO: Run the main loop for the chatbot
head_agent = Head_Agent()
head_agent.main_loop()
