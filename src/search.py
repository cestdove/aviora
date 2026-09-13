# search.py
# Audio Search Script

import os 
import soundfile as sf
import librosa
import numpy as np 
import chromadb 
from audio_utils import extract_features_from_signals


def query_acoustic_database(query_vector, n_results=5):
    """Does geometric search on ChromaDB starting from a calculated vector."""
    
    # Initialize the local database on disk
    print("")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("########## ChromaDB Search Process #############")
    print("------------------------------------------------")
    print("Searching ChromaDB databases...")
    print("")

    # Check if the ChromaDB database exists
    if not os.path.exists("data/chroma_db"):
        print("ChromaDB database not found. Please run the ingestion process first to create the database.")
        print("")
        print("Exiting the search process...")
        print("------------------------------------------------")
        return
    else:
        print("ChromaDB database found. Proceeding with the search process.")
        print("-----------------------------------------------")

        # Connect to the existing ChromaDB database on read-only mode
        print("Connecting to ChromaDB...")
        print("")
        client = chromadb.PersistentClient(path="data/chroma_db")
        print("Connected to ChromaDB successfully.")
        print("------------------------------------------------")
        
        print("Retrieving the collection from ChromaDB...")
        collection = client.get_collection(name="audio_features")
        print("")
        print("Collection 'audio_features' found in ChromaDB.")
        print("You can now perform similarity searches using the query audio features.")
        print("------------------------------------------------")

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        return results

    
    
if __name__ == "__main__":
    print("Search script started. Run streamlit app.py")