import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Index configuration
INDEX_NAME = os.getenv("PINECONE_INDEX", "support-rag")
EMBEDDING_DIMENSION = 1024

# Check if index already exists
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"Index '{INDEX_NAME}' created successfully!")

else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Verify index configuration
index_info = pc.describe_index(INDEX_NAME)

print("Index name:", index_info.name)
print("Dimensions:", index_info.dimension)
print("Metric:", index_info.metric)
print("Status:", index_info.status)