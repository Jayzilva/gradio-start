from fastapi import FastAPI, Query
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from dotenv import load_dotenv
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from traceloop.sdk import Traceloop

load_dotenv()
os.environ['OPENAI_API_KEY'] = "OPEN-API-KEY"

Traceloop.init(
  disable_batch=True, 
  api_key="TRACELOOP-API-KEY",
)

app = FastAPI()

def load_documents():
    try:
        documents = SimpleDirectoryReader("documents").load_data()
        return documents
    except Exception as e:
        return {"error": str(e)}

documents = load_documents()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

@app.get("/query")
def query_documents(query: str = Query(..., description="Enter your query")):
    response = query_engine.query(query)
    return {"response": str(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
