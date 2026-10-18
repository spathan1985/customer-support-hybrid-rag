import json
from collections import Counter

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))

print("\nDocument types:")
print(Counter(c["document_type"] for c in chunks))

print("\nFAQ chunks:")
faq_chunks = [
    c for c in chunks
    if "faq" in c["source"].lower()
]

print("FAQ chunks indexed:", len(faq_chunks))

for chunk in faq_chunks[:3]:
    print(chunk["source"])
    print(chunk["text"][:150])

for chunk in chunks:
    if chunk["document_type"] == "data":
        print("Source:", chunk["source"])
        print("Text:", chunk["text"][:300])