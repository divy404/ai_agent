import json
import os

from dotenv import load_dotenv
import numpy
from openai import OpenAI


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)
EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

#Load the documents first
def load_documents(folder=KB_FOLDER):

    documents = []

    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".txt"):
            continue

        path = os.path.join(folder,file_name)
        with open(path,"r",encoding="utf-8") as f:
            documents.append(
                {
                    "source":file_name,
                    "text":f.read()
                }
            )
    return documents


#get embeddings of a text from embedding model
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

#get cosine similarity between two vectors 
def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod/(norm_a*norm_b)

#add embeddings in the chunks
def add_embeddings(chunks):
    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        chunk["vector"] = vector
    return chunks

#Search method to search the top k documents for the user query
def search(query,chunks,top_k=3):
    query_vector = get_embedding(query)

    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append(
            {
                "score":float(score),
                "source":chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "text":chunk["text"],
                "size":len(chunk["text"])
            }
        )
    results.sort(key=lambda rec:rec["score"],reverse=True)
    return results[:top_k]

#creating chunks
def create_chunks():
    documents = load_documents()
    chunks=[]
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks

#chunk document
def chunk_document(document):
    chunks=[]
    chunk_id = 0
    for part in document["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append({
                "source":document["source"],
                "chunk_id":chunk_id,
                "text":part
            })
            chunk_id+=1
    return chunks

#testing method
def test(user_query,chunks,top_k=3):
    print(f"\n User query: {user_query}\n")

    results = search(user_query,chunks,top_k)

    for r in results:
        print(f"{r['score']:.2f}-{r['source']}-{r['size']} characters)")
        print(f"{r['text']}\n")

def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE,"w",encoding="utf-8") as f:
        json.dump(chunks,f,indent=4)

def load_vector_store():
    with open(VECTOR_STORE_FILE,"r",encoding="utf-8") as f:
        return json.load(f) #returns the chunks

#Lets start testing
chunks = []
if os.path.exists(VECTOR_STORE_FILE):
    chunks=load_vector_store()
    print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE_FILE}")
else:
    chunks = create_chunks()
    chunks=add_embeddings(chunks)
    save_vector_store(chunks)
    print(f"Created and embedded {len(chunks)} chunks and saved to the vector store file {VECTOR_STORE_FILE}")



print(f"we have embedded{len(chunks)} chunks from our knowledge base")

user_query = "What are the benefits of maternity leave?"
test(user_query,chunks,3)
