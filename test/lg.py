import os
import json
import requests
from decimal import Decimal

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
from langchain.schema import Document

from langchain_core.runnables import RunnableSequence

from langchain_experimental.agents import create_csv_agent

#from langchain_cohere import create_csv_agent

import pandas as pd


# Load environment variables
load_dotenv()

# Define global interaction context ???????????????
interaction_context = {
    "client": {},
    "order": {}
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

# Load documents from Excel
def load_excel(file_path, output_path):
    try:
        data = pd.read_excel(file_path, engine= 'openpyxl')
        data.to_csv(output_path, index= False)
        return data
    
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return []

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
def main_chain():
    # Initialize the LLM
    main_llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )

    # Define the custom prompt template
    custom_prompt = PromptTemplate(
        template=(
            "You are a helpful assistant with concise and clear answers. "
            "Your primary task is to correct any grammatical mistakes in the {input_text}. "
            "Your response must always be in the following JSON format:\n"
            "{{\"user_input\": \"{input_text}\", \"extractor\": \"<PROMPT>\"}}\n"
            "Where:\n"
            "- \"user_input\" contains the corrected version of {input_text} with grammatical mistakes fixed.\n"
            "- \"extractor\" is replaced with the appropriate prompt based on the context of the {input_text} as follows:\n\n"
            "If the {input_text} contains user information (e.g., name or email):\n"
            "Replace <PROMPT> with:\n"
            "  'Respond in a JSON format: {{\"name\": \"<NAME>\", \"email\": \"<EMAIL>\"}}, use double quotes always for each element inside of the JSON.'\n\n"
            "If the {input_text} is about a product order:\n"
            "Replace <PROMPT> with:\n"
            "  'Provide details of the product of the <ORDER> based on the CSV data available to you, you must address the quantity of each product in {input_text}. \n "
            "If the order contains more than one product, create more entries to the array, where each entry is in the format that will follow.\n"
            "Compute <value> as the multiplication of <quantity> and the unit value from the CSV, quantity must be an integer and value must be a float. \n"
            "Respond in the following format, use double quotes always for the elements of the JSON as it shows the following format: "
            "[{{\"product_id\": \"<product_id>\", \"product_name\": \"<product_name>\", \"quantity\": <quantity>, \"value\": <value>}}]. '\n\n"
            "Important:\n"
            "- Do not modify any values within the dictionary of the order.\n"
            "- Always adhere strictly to the JSON format described.\n"
            "- Only return the JSON output; do not include any additional information or explanations."
   
        ),
        input_variables=["input_text"]
    )

    # Create the chain using a RunnableSequence
    chain = custom_prompt | main_llm

    return chain

#Extraction LLM chain
def extraction_chain(csv_file_path):

    # Create the LLM instance
    extraction_llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )

    # Create the CSV agent with the custom prompt
    agent = create_csv_agent(
        llm=extraction_llm,
        path=csv_file_path,
        verbose=True,
        allow_dangerous_code=True
        #prompt=custom_prompt  # Attach the custom prompt here
    )

    return agent

# Create the main chain
def front_end_chain(vectorstore):
    main_llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()

    custom_prompt = PromptTemplate(
        template=(
            "You are a helpful assistant with concise and clear answers no more than 30 characters."
            "The answers must be in the following format:"
            " "
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

def extraction_validation(extracted_data):
    """
    Validate and assign values from either the client or order fields to interaction context.
    Adds a `position` key to each order, which contains the 1-based index of the order item.
    
    Args:
        extracted_data (dict or list): Extracted data from the response, either a dictionary for client
                                       or a list of dictionaries for orders.
    
    Returns:
        dict: Updated interaction context containing both client and order details.
    """
    global interaction_context

    try:
        print(isinstance(extracted_data, list))

        if isinstance(extracted_data, dict):
            # Validate and update client information
            if "client" in extracted_data:
                client_info = extracted_data["client"]
                # Validate client fields using Pydantic
                user_info = UserInput(**client_info)
                # Update interaction context with validated client data
                interaction_context["client"] = {
                    "name": user_info.name,
                    "email": user_info.email
                }

        elif isinstance(extracted_data, list):
            # Validate and update order details
            order_details = []
            for index, order in enumerate(extracted_data):
                product_id = order.get("product_id")
                product_name = order.get("product_name")
                quantity = order.get("quantity")
                value = order.get("value")
                print(value)
                # Ensure all required fields are present
                if product_id and product_name and quantity and value:
                    order_details.append({
                        "order_position": index + 1,  # Add 1-based index
                        "product_id": product_id,
                        "product_name": product_name,
                        "quantity": quantity,
                        "value": value
                        
                    })
            # Update interaction context with order details
            interaction_context["order"] = order_details

        return interaction_context

    except Exception as e:
        print(f"Error during validation: {e}")
        return interaction_context

# Store user chains and memory
user_storage = {}#CHANGE

def user_chain(): #CHANGE
    

    return 

# Main RAG system
def agent_system(csvfile_path):
    main_chain_instance= main_chain()
    extraction_chain_instance= extraction_chain(csvfile_path)

    
    return  main_chain_instance, extraction_chain_instance


def interaction_system(user_input, main_chain, extraction_chain):

    raw_response= json.loads(main_chain.invoke({"input_text": user_input}).content)
    
    raw_extraction_response= extraction_chain.invoke(str(raw_response['extractor']))
    extraction_response= json.loads(raw_extraction_response["output"])

    extraction_validation(extraction_response)


#    chain, memory = user_chain(user_ID)
#    chain_invoke = chain.invoke({
#        "question": user_input,
#        "chat_history": memory.load_memory_variables({})["chat_history"]
#    })

#    return {
#        "answer": chain_invoke,
#        "extracted_info": extracted_info
#    }

# Initialize documents and vectorstore for main LLM
file_path = "data/03_ServiceDemo_Menu.pdf"
documents = load_document(file_path)
main_vectorstore = setup_vectorstore(documents)

# Initialize documents and vectorstore for secondary LLM
xlfile_path = "data/04_ServiceDemo_Items.xlsx"
csvfile_path = "data/04_ServiceDemo_Items.csv"
csv_document = load_excel(xlfile_path, csvfile_path)


if __name__ == "__main__":
    user_ID = "user123"

    print("Welcome! You can start interacting with the assistant.")
    print("Type 'exit' to end the conversation.\n")

    main_chain, extraction_chain= agent_system(csvfile_path)

    while True:
        # User input
        #chain = main_chain()

        #extraction = extraction_chain(csvfile_path)
        

        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Thank you for using the assistant. Goodbye!")
            break
        
        
        
        interaction_system(user_input, main_chain, extraction_chain)

        print(interaction_context)

        # Post the order data to the Flask API
        if len(interaction_context["order"]) > 0:
            
            orders= interaction_context.get("order", [])
            print("Posting order data to the API...")
            for index, order in enumerate(orders):
                print(order)
                try:
                    
                    response = requests.post(
                        "http://127.0.0.1:4000/api/flask/order_items",  # Adjust port if necessary
                        json= order
                    )
                    print("API Response:", response.json())
                except Exception as e:
                    print(f"Error posting data to API: {e}")

        
        
        
        # Process the input using the RAG system
        #response = RAGsystem(user_ID, user_input)

        # Print the answer and extracted information
        #print("\nAssistant:", response["answer"]["answer"])
        #print("Extracted Info:", response["extracted_info"])
        #print("-" * 50)
        

        #raw_response= chain.invoke({"input_text": user_input}) #.content
        #raw_response_content= raw_response.content
        #json_raw_response= json.loads(raw_response_content)
        

        #print(raw_response)
        #print(raw_response_content)
        #print("Response:", json_raw_response['extractor'])

        #extraction_response= extraction.invoke(str(json_raw_response['extractor']))
        #print("Extraction:", extraction_response["output"])

        #print("extraction_response['output'][0]:", extraction_response["output"][0])
        #json_extraction= json.loads(extraction_response["output"])
        #print('json_extraction[0]: ', json_extraction[0])