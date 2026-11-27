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

#add embeddings in the document
def add_embeddings(documents):
    for document in documents:
        vector = get_embedding(document["text"])
        document["vector"] = vector
    return documents

#Search method to search the top k documents for the user query
def search(query,documents,top_k=3):
    query_vector = get_embedding(query)

    results = []
    for document in documents:
        score = cosine_similarity(query_vector, document["vector"])
        results.append(
            {
                "score":float(score),
                "source":document["source"],
                "text":document["text"],
                "size":len(document["text"])
            }
        )
    results.sort(key=lambda rec:rec["score"],reverse=True)
    return results[:top_k]

#testing method
def test(user_query,documents,top_k=3):
    print(f"\n User query: {user_query}\n")

    results = search(user_query,documents,top_k)

    for r in results:
        print(f"{r['score']:.2f}-{r['source']}-{r['size']} characters)")

#Lets start testing
documents = load_documents()
documents=add_embeddings(documents)

print(f"we have embedded{len(documents)} documents")

user_query = "How many days will i get for maternity leave?"
test(user_query,documents,3)
