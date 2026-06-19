# app.py
# Streamlit App for BioAcoustic Vector Search Search Engine

import os
import streamlit as st 
import chromadb 
import soundfile as sf 
import librosa 
import numpy as np 
import librosa.display
import librosa.display
import matplotlib.pyplot as plt
from audio_utils import extract_features_from_signals, slice_signal_into_chunks
from search import query_acoustic_database


# Streamlit page configuration
st.set_page_config(page_title="BioAcoustic Vector Search", page_icon=":musical_note:", layout="wide") 
st.title("BioAcoustic Vector Search :musical_note:")
st.subheader("Upload an audio file to find similar audio files in the database.")

# User uploader audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Show an audio player for the uploaded file
    st.audio(uploaded_file, format='audio/wav')
    st.info("Doing signal analysis and querying ChromaDB...")

    # Extract features from the uploaded audio file and perform the search
    st.info("Extracting features from the uploaded audio file and searching for similar audio files in the database...")

    try: 
        # Reads the file from the memory 
        y, sr = sf.read(uploaded_file)
        
        if y is None or sr is None:
            raise ValueError(f"Could not read the audio file: {uploaded_file.name}")
        
        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)

        # Manually cuts the audio to the first 10s to be equal to ingest.py
        max_length = 10 * sr
        y = y[:max_length]

        # Cut the fragments/chunks into 3 seconds
        query_chunks = slice_signal_into_chunks(y, sr, chunk_duration=3)
        
        if not query_chunks:
            raise ValueError("L'audio caricato è troppo corto o contiene solo silenzio.")

        # Takes the first useful fragment and does the classical research
        query_vector = extract_features_from_signals(query_chunks[0], sr)
        results = query_acoustic_database(query_vector, n_results=5)

        # Display the search results
        st.success("Search completed! Here are the top similar audio files found in the database:")

        # Iterate through the search results and display the titles and audio players for each similar audio file
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                track_id = results["ids"][0][i]
                distance = results["distances"][0][i]

                # First define metadatas by extracting them from ChromaDB
                if results.get("metadatas") and results["metadatas"][0] is not None:
                    metadata = results["metadatas"][0][i]
                else:
                    metadata = None


                # Then use them safely because the variable exists
                title = metadata.get("title", f"Track {track_id}") if metadata else f"Track {track_id}"
                file_path = metadata.get("file_path", None) if metadata else None

                with st.container():
                    st.markdown(f"### {i+1}. {title}")
                    st.text(f"Euclidean Distance: {distance:.4f}")

                    if file_path and os.path.exists(file_path):
                        # Audio preview
                        with open(file_path, "rb") as f:
                            st.audio(f.read(), format="audio/wav")
                        
                        # Spectrogram
                        with st.expander("Show Spectrogram"):
                            fig, ax = plt.subplots(figsize=(10, 3))
                            
                            # Graph generation
                            y_res, sr_res = librosa.load(file_path, sr=None)
                            S = librosa.feature.melspectrogram(y=y_res, sr=sr_res, n_mels=128, fmax=8000)
                            S_dB = librosa.power_to_db(S, ref=np.max)
                            
                            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_res, fmax=8000, ax=ax, cmap='magma')
                            fig.colorbar(img, ax=ax, format='%+2.0f dB')
                            
                            st.pyplot(fig)
                            plt.close(fig) # Frees the RAM
                    else: 
                        st.warning("Source file not found on disk.")
                    st.markdown("---")
        else:
            st.warning("No match found in the database.")

    except Exception as e:
        st.error(f"There was a connection error between the modules: {e}")


