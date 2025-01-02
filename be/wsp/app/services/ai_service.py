
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import requests

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define global interaction context
interaction_context = {
    "name": None,
    "email": None
}

# Define Pydantic model for extracted data
class UserInput(BaseModel):
    name: Optional[str] = Field(None, description="The user's name")
    email: Optional[EmailStr] = Field(None, description="The user's email address")

# Load documents
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Setup vectorstore
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Create the main chain
def create_chain(vectorstore):
    main_llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()

    custom_prompt = PromptTemplate(
        template=(
            "You are a helpful assistant with concise and clear answers no more than 30 characters. "
            "Expand if asked for more details. Respond in the language of the user's query. "
            "Say hello on first contact. Answer as 'we' if the question is about NextGen Predictions. "
            "Use the following conversation history:\n{chat_history}\n. "
            "Now answer the user's latest question based on the context and conversation."
            "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
        input_variables=["context", "question", "chat_history"]
    )

    memory = ConversationBufferMemory(
        llm=main_llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=main_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return chain, memory

# Corrected extraction LLM chain
def create_extraction_chain():
    extraction_llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )

    extraction_prompt = PromptTemplate(
        template=(
            "You are a variable identifier agent, your purpose is to identify variables within a user's prompt, input will be either in spanish or english."
            "The user's input is this one:"
            "\n'{input_text}'\n"
            "The variables you need to identify are the name and the email of the user."
            "I you don't find any variable use `null` value as an attribute"
            "For the name format, fix any grammatical mistake in the typing"
            "The answer you give must be a JSON-like format:\n"
            "{{\"name\": \"<NAME>\", \"email\": \"<EMAIL>\"}}.\n"
            "If any value is missing, use `null`as an attribute, not as a string"
            "don't give any additional data, just the JSON format."
        ),
        input_variables=["input_text"]
    )

    def extract_from_llm(input_text):
        prompt_text = extraction_prompt.format(input_text=input_text)
        response = extraction_llm.invoke(prompt_text)

        # Extract content if the response contains it
        extracted_content = response.content if hasattr(response, "content") else "{}"
        print("Cleaned Extraction Content:", extracted_content)  # Debugging log
        return extracted_content

    return extract_from_llm

# Base URL of your Flask app
api_url = "http://dbflaskapp:4000/api/flask/users"

# Extract user information and update context
def extract_user_info(user_input):
    extract_fn = create_extraction_chain()
    raw_response = extract_fn(user_input)

    try:
        # Validate and parse using Pydantic
        user_info = UserInput.model_validate_json(raw_response)
        print("Parsed Info:", user_info.model_dump())  # Debugging log

        # Update interaction context
        global interaction_context
        if user_info.name:
            interaction_context["name"] = user_info.name
        if user_info.email:
            interaction_context["email"] = user_info.email
        
        if interaction_context["name"] != None and interaction_context["email"] != None:
            response = requests.post(api_url, json=interaction_context)

            # Print the response
            if response.status_code == 201:
                print("User created successfully:", response.json())
            else:
                print("Error creating user:", response.json())

        print("Interaction Context: ", interaction_context)
        return interaction_context

    except Exception as e:
        print(f"Extraction Error: {e}")
        return interaction_context

# Initialize documents and vectorstore
#file_path = "../../data/NEXTGENPREDICTIONS.pdf"
file_path = os.path.join(os.path.dirname(__file__), "../../data/NEXTGENPREDICTIONS.pdf")

# Normalize the path
absolute_file_path = os.path.abspath(file_path)

documents = load_document(file_path)
vectorstore = setup_vectorstore(documents)

# Store user chains and memory
user_storage = {}

def user_chain(user_ID):
    if user_ID not in user_storage:
        chain, memory = create_chain(vectorstore)
        user_storage[user_ID] = {"chain": chain, "memory": memory}

        # Limit user storage
        if len(user_storage) > 20:
            oldest_user_id = next(iter(user_storage))
            print(f"Removing oldest user: {oldest_user_id}")
            del user_storage[oldest_user_id]

    return user_storage[user_ID]["chain"], user_storage[user_ID]["memory"]

# Main RAG system
def RAGsystem(user_ID, user_input):
    extracted_info = extract_user_info(user_input)
    chain, memory = user_chain(user_ID)
    chain_invoke = chain.invoke({
        "question": user_input,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    return {
        "answer": chain_invoke,
        "extracted_info": extracted_info
    }



def generate_response(message, wa_id, name):
    logging.info(f"Conversation with {name}, wa_id: {wa_id}")

    # Run the assistant and get the new message
    AIresponse = RAGsystem(wa_id, message)
    new_message = AIresponse["answer"]["answer"]

    return new_message
