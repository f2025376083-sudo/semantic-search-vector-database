# Semantic Search Using a Vector Database

**Author:** Ayesha

## Project Overview

This project demonstrates how a vector database can be used for semantic search. Ten university FAQ records are converted into numerical embeddings using the `all-MiniLM-L6-v2` embedding model. The embeddings, original text, and metadata are stored in a local Chroma vector database.

Users can enter natural-language questions. The question is converted into an embedding and Chroma searches for stored vectors with similar meaning.

## Technologies

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- Vector embeddings
- Semantic similarity search

## Workflow

Information → Embedding Model → Vector → Chroma Vector Database → Similarity Search → Relevant Results

## Metadata

Each record contains:
- Category
- Title

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

The first run may download the embedding model.

## Example Queries

1. How can I get help if my university Wi-Fi is not working?
2. What financial assistance is available for students?
3. Where can I find information about my exam schedule?
4. How do I pay my university semester charges?

## Expected Result

The system should return relevant FAQ records even when the wording of the query is different from the wording stored in the database. For example, a query about "Wi-Fi not working" should retrieve the IT support record.

## Conclusion

This project demonstrates the basic role of a vector database in semantic search. Instead of matching only exact keywords, the system compares vector representations and retrieves information with similar meaning. This approach is commonly used in knowledge bases, recommendation systems, and Retrieval-Augmented Generation (RAG) applications.
