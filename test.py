import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Nebius client using OpenAI SDK
client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url="https://api.tokenfactory.nebius.com/v1/"
)

# Generate embeddings using Nebius
response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input="My thermostat is not connecting to Wi-Fi.",
    dimensions=1024
)

# Extract the embedding vector
embedding = response.data[0].embedding

# Verify dimensions
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])