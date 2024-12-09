import os
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_vector_store():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_name = "medical-chatbot"

    # Initialize and return PineconeVectorStore
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )


def init_vector_store(texts):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_name = "medical-chatbot"

    # Check if the index exists, create if it doesn't
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension for all-MiniLM-L6-v2
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
        vectors_exist = False
    else:
        print(f"Index {index_name} already exists.")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        vectors_exist = stats.total_vector_count > 0

    # Initialize PineconeVectorStore
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    # Upload vectors if they don't exist
    if not vectors_exist:
        print("Uploading vectors to Pinecone...")
        vector_store.add_documents(texts)
    else:
        print("Vectors already exist in Pinecone. Using existing index...")

    return vector_store


def get_or_create_vector_store(texts):
    return init_vector_store(texts)
