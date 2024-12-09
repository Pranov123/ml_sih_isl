import os
from dotenv import load_dotenv
from pdf_processor import process_pdf
from vector_store import get_vector_store
from chatbot import init_chatbot, chat


def main():
    load_dotenv()
    
    vector_store = get_vector_store()

    chain = init_chatbot(vector_store)

    chat(chain)


if __name__ == "__main__":
    main()