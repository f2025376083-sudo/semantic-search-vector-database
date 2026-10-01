# Semantic Search using a Vector Database
# Author: Ayesha

from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load an embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create a local Chroma vector database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="university_faqs")

# 3. Sample information (10 short university FAQs)
documents = [
    "Students can apply for admission online through the university admission portal.",
    "The computer science department offers courses in artificial intelligence, machine learning, and software engineering.",
    "Students can pay their semester fees through the university online payment portal or designated bank.",
    "The library is open from 8 AM to 8 PM on weekdays and provides access to digital and printed resources.",
    "Students can contact the IT support office for problems related to university accounts, Wi-Fi, and computer systems.",
    "The university provides scholarships based on academic performance and financial need.",
    "Students can check their examination timetable through the student portal before the examination period.",
    "Attendance requirements vary by course, but students are expected to attend classes regularly.",
    "Students can register for courses through the online student information system during the registration period.",
    "The career office helps students with internships, CV preparation, interview skills, and job opportunities."
]

metadatas = [
    {"category": "admission", "title": "Online Admission"},
    {"category": "academics", "title": "Computer Science Programs"},
    {"category": "fees", "title": "Semester Fees"},
    {"category": "library", "title": "Library Hours"},
    {"category": "IT support", "title": "Technical Support"},
    {"category": "scholarships", "title": "Scholarships"},
    {"category": "exams", "title": "Examination Timetable"},
    {"category": "attendance", "title": "Attendance Policy"},
    {"category": "registration", "title": "Course Registration"},
    {"category": "career", "title": "Career Office"}
]

ids = [f"faq_{i}" for i in range(1, len(documents) + 1)]

# 4. Convert information into embeddings
embeddings = model.encode(documents, normalize_embeddings=True)

# 5. Store vectors, documents, and metadata in Chroma
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings.tolist()
)

print("\n" + "=" * 70)
print("VECTOR DATABASE SEMANTIC SEARCH SYSTEM")
print("=" * 70)
print(f"Stored records: {collection.count()}")

# 6. Semantic search function
def semantic_search(question, n_results=3):
    query_embedding = model.encode(question, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print("\n" + "-" * 70)
    print(f"QUERY: {question}")
    print("-" * 70)

    for i, (doc, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):
        print(f"\nResult {i}")
        print(f"Title: {metadata['title']}")
        print(f"Category: {metadata['category']}")
        print(f"Distance: {distance:.4f}")
        print(f"Information: {doc}")

# 7. Test at least three natural-language queries
semantic_search("How can I get help if my university Wi-Fi is not working?")
semantic_search("What financial assistance is available for students?")
semantic_search("Where can I find information about my exam schedule?")
semantic_search("How do I pay my university semester charges?")
