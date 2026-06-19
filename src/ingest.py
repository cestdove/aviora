# ingest.py
# Audio Ingestion Script

import os 
import pandas as pd
import numpy as np 
import chromadb
import librosa
from audio_utils import extract_features_from_signals, slice_signal_into_chunks



def main():
    """Main function."""

    # Load the CSV file containing the audio catalog
    csv_file_path = "data/audio_catalog.csv"

    # Check if the CSV file exists
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    print("")
    print("------------------------------------------------")
    print("CSV file search process...")
    print("")
    print("CSV file found. Proceeding with the ingestion process.")
    print("------------------------------------------------")
    print("")

    # Initialize the local database on disk 
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("########## ChromaDB Ingestion Process ##########")
    print("------------------------------------------------")
    print("Connecting to ChromaDB...")
    print("")

    # ------ Control over the ChromaDB rewrite ------
    rewrite = True  # Set to True to rewrite the entire database

    if rewrite:
        print("There is already an existing ChromaDB database.")
        if os.path.exists("data/chroma_db"):
            import shutil
            shutil.rmtree("data/chroma_db")  # Remove the existing database directory
            print("Existing ChromaDB database removed.")
            print("Proceeding to create a new ChromaDB database...")
            print("")
            print("New ChromaDB database created successfully.")
        else:
            print("No existing ChromaDB database found. Proceeding to create a new one.")
    # ------------------------------------------------

    print("------------------------------------------------")
    print("Initializing the ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")  # specify the path to the local database
    print("")
    print("ChromaDB client initialized successfully.")
    print("------------------------------------------------")

    # Create a collection in the database to store the audio features 
    print("Creating a collection in ChromaDB...")
    collection = chroma_client.create_collection(name="audio_features", metadata={"hnsw:space": "l2"}) # specify the L2 distance metric for the HNSW index

    print("")
    print("Collection created successfully.")

    print("------------------------------------------------")

    # Read the CSV file into a pandas DataFrame
    print("Reading the CSV file...")
    df = pd.read_csv(csv_file_path)
    print("")
    print(f"CSV file read successfully. Number of records: {len(df)}")

    print("------------------------------------------------")

    print("Starting the ingestion process for audio files...")
    print("")

    # Iterate through each row in the DataFrame and process the audio files

    success_count = 0 
    error_count = 0

    for index, row in df.iterrows():
        audio_id = str(row['id'])  # Ensure the ID is a string
        title = row['title']
        file_path = row['file_path']

        print(f"Processing audio file: {file_path} (ID: {audio_id}, Title: {title})")

        try: 
            # Opens the file, extracting the acoustic signal y and the frequency sr
            y, sr = librosa.load(file_path, sr=None, duration=10)
            
            # 2. Divide l'audio in frammenti (chunks) da 3 secondi rimuovendo i silenzi
            # Divides the audio signal into 3s multiple fragments (chunks), removing the silences
            chunks = slice_signal_into_chunks(y, sr, chunk_duration=3)
            
            if not chunks:
                print(f"  [!] Traccia troppo corta o di solo silenzio: {file_path}. Salto.")
                continue
                
            # Iterates over the fragments/chunks and saves them gradually into ChromaDB
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{audio_id}_chunk_{chunk_idx}"
                vectorized = extract_features_from_signals(chunk, sr)

                collection.add(
                    embeddings=[vectorized],
                    metadatas=[{"title": title, "file_path": file_path, "chunk_index": chunk_idx}],
                    ids=[chunk_id]
                )
            
            print(f"Successfully added {len(chunks)} chunks to ChromaDB: {file_path} (ID: {audio_id})")
            success_count += 1
            print("")

        except Exception as e:
            print(f"Error processing audio file {file_path} (ID: {audio_id}): {e}")
            error_count += 1    # increment the error count for each failed ingestion
            # remove the entry from the collection if there was an error
            collection.delete(ids=[audio_id])
            print("Assure that the audio file exists and is in a supported format (e.g., WAV, MP3).")
            print("The file will be skipped to avoid halting the ingestion process.")
            print("")

            # break  # Stop processing if there's an error to prevent further issues
            
    if error_count == 0:
        print("------------------------------------------------")
        print("Ingestion process completed successfully.")
        print(f"Total records ingested: {success_count}")
        print("You can now use the ChromaDB database for audio retrieval and analysis.")

    else:
        print("------------------------------------------------")
        print("Ingestion process completed with some errors.")
        print(f"Total records processed: {success_count + error_count}")
        print(f"Successfully ingested records: {success_count}")
        print(f"Records with errors: {error_count}")
        print("Please review the error messages above for details on any issues encountered.")
        print("You can still use the ChromaDB database, but some audio files may not have been ingested.")

    print("------------------------------------------------")

if __name__ == "__main__":
    main()