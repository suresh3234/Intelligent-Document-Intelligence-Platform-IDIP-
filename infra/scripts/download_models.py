import os
import sys

def download_models():
    # Enforce downloading to the shared mount/cache folder
    os.environ["HF_HOME"] = "/app/models"
    os.environ["TRANSFORMERS_CACHE"] = "/app/models"
    
    print("Pre-downloading model weights for offline inference...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("1. Downloading BAAI/bge-large-en-v1.5 embeddings model...")
        SentenceTransformer("BAAI/bge-large-en-v1.5")
    except Exception as e:
        print(f"Error downloading embeddings model: {e}")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("2. Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 reranker model...")
        AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"Error downloading reranker model: {e}")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        print("3. Downloading dslim/bert-base-NER named entity recognition model...")
        AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    except Exception as e:
        print(f"Error downloading NER model: {e}")
        sys.exit(1)

    print("All weights downloaded successfully and cached in /app/models!")

if __name__ == "__main__":
    download_models()
