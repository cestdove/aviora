# birds.py
# Making DB and CSV from Xeno-Canto for birds chants

import os
import time 
import requests 
import pandas as pd

def build_european_birds_dataset(samples_per_bird=40):
    # Dictionary of some common species in Scientific Latin and Italian
    target_birds = {
        # Original ones
        "Passer italiae": "Passero",
        "Turdus merula": "Merlo",
        "Parus major": "Cinciallegra",
        "Streptopelia decaocto": "Tortora", 
        
        # Added
        "Erithacus rubecula": "Pettirosso",
        "Fringilla coelebs": "Fringuello",
        "Sylvia atricapilla": "Capinera",
        "Carduelis carduelis": "Cardellino",
        "Luscinia megarhynchos": "Usignolo",
        "Troglodytes troglodytes": "Scricciolo",
        "Alauda arvensis": "Allodola"
    }

    audio_dir = "data/audio_files"  
    os.makedirs(audio_dir, exist_ok=True)
    csv_data = []

    print("####################### Bioacoustic Dataset Generator #######################")
    print("-----------------------------------------------------------------------------")
    print(f"Target species: {len(target_birds)} - Samples per Species {samples_per_bird}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # API key - read from environment, never hardcode it in source
    XENO_CANTO_API_KEY = os.environ.get("XENO_CANTO_API_KEY")
    if not XENO_CANTO_API_KEY:
        raise RuntimeError(
            "Set the XENO_CANTO_API_KEY environment variable before running this script "
            "(e.g. export XENO_CANTO_API_KEY=your_key)."
        )

    for scientific_name, italian_name in target_birds.items():
        print(f"\nResearching about: {italian_name} ({scientific_name})...")
        
        # Official endpoint v3
        api_url = "https://xeno-canto.org/api/3/recordings"

        # v3 requires field tags (gen:, sp:, cnt:, q:, ...) for every search
        # term - it no longer accepts bare free text like "Passer italiae".
        genus, _, species = scientific_name.partition(" ")
        query_str = f"gen:{genus} sp:{species} q:A cnt:Italy"

        try:
            # Using only User-Agent to not get blocked by anti-bot systems
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

            # Let requests build/encode the query string instead of doing it by hand
            response = requests.get(
                api_url,
                params={"query": query_str, "key": XENO_CANTO_API_KEY},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status() 
            data = response.json()
            
            recordings = data.get('recordings', [])
            print(f"  [OK] Connection was successful. Found {len(recordings)} tracks.")
            
            
        except Exception as e:
            print(f"  [!] Failed connection for {italian_name}: {e}. Skip.")
            continue

        if not recordings:
            print(f" [!] No high quality registration found for {italian_name}.")
            continue 

        print(f"Found {len(recordings)} valid tracks. Downloading the first {samples_per_bird}...")

        downloaded = 0
        for rec in recordings:
            if downloaded >= samples_per_bird:
                break

            track_id = str(rec.get("id", "unknown"))
            file_url = rec.get("file")
            if not file_url: 
                continue

            # Xeno-canto sometimes returns protocol-relative URLs (e.g. "//xeno-canto.org/...")
            if file_url.startswith("//"):
                file_url = "https:" + file_url

            # Create clean name file using italian name
            safe_title = italian_name.replace(" ", "_")
            file_path = os.path.join(audio_dir, f"{safe_title}_{track_id}.mp3")

            try:
                # Download the actual audio file, not the search API endpoint
                audio_response = requests.get(file_url, headers=headers, timeout=10)
                audio_response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(audio_response.content)

                if os.path.getsize(file_path) < 1024:
                    raise ValueError("The downloaded file is empty or corrupted.")

                csv_data.append({
                    "id" : track_id,
                    "title" : italian_name,
                    "file_path" : file_path
                })
                downloaded += 1
                time.sleep(0.5) # rate limit protection

            except Exception as e:
                print(f"    There was an error downloading the following file: {track_id}")
                print(f"{e}")
                if os.path.exists(file_path):
                    os.remove(file_path)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = "data/audio_catalog.csv"
            df.to_csv(csv_path, index=False)
            print("\n" + "="*45)
            print("Dataset has been completed successfully.")
            print(f"There are {len(df)} catalogued tracks in total into {csv_path}.")
        else: 
            print("")
            print("[!] Error: No track has been downloaded.")



if __name__ == "__main__":
    build_european_birds_dataset(samples_per_bird=40)