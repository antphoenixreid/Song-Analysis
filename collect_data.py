import spotipy
from spotipy.oauth2 import SpotifyOAuth

import os

from pytubefix import YouTube, Search
import numpy as np
import pandas as pd

from lyricsgenius import Genius

# Set up Spotify API Access
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="638be10d0e7f44a091b815f77ed280e6",
#                                                client_secret="ff842693b14d445cae25161a8ca77972",
#                                                redirect_uri="http://127.0.0.2:9090",
#                                                scope="user-library-read"))

# Set up Genius API Access
genius = Genius('zCgrhGOfN0ejD7ezJrfrjKQ-zmEj78-Hn_Otx_LsYBeiSqG8-mvnPwoMO1A-tylA')

def is_substring_in_string(substring, full_string):
    # Split both strings into sets of words (ignoring case)
    substring_set = set(substring.lower().split())
    full_string_set = set(full_string.lower().split())

    # Check if the substring set is a subset of the full string set
    return substring_set.issubset(full_string_set)

# Read song list from CSV
song_list = pd.read_csv('dataset/Song_List.csv')
tracks = song_list['Song Title'].tolist()
artists = song_list['Artist'].tolist()
yt_ids = song_list['YouTube ID'].tolist()

# Parse Through Songs to Collect Data
for i in range(len(tracks)):
    track_artist = artists[i]
    track_name = tracks[i]
    search_kw = track_artist + ' - ' + track_name

    print('Processing ' + search_kw)
    if pd.notna(yt_ids[i]):
        video_id = yt_ids[i]
    else:
        s = Search(search_kw)
        video_id = s.results[0].video_id  # Default to first result

    try:
        # Extract the audio to download
        selected_video = YouTube(f'https://www.youtube.com/watch?v={video_id}').streams.filter(only_audio=True).first()
        output_path = 'D:/Engineering/Signal Processing/Personal Projects/Song Analysis/dataset/songs/'
        print(f'Downloaded: {track_artist} - {track_name} from YouTube ID: {video_id}')
    except Exception as e:
        print(f'Failed to download: {track_artist} - {track_name} from YouTube ID: {video_id}. Error: {e}')
        continue

    folder_path = os.path.join(output_path, search_kw)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    selected_video.download(output_path=folder_path, filename=search_kw + '.mp3')
    print(f'Saved to {folder_path}/{search_kw}.mp3')

    # Fetch lyrics using Genius API
    try:
        song = genius.search_song(track_name, track_artist)
        if song and song.lyrics:
            with open(os.path.join(folder_path, 'lyrics.txt'), 'w', encoding='utf-8') as f:
                f.write(song.lyrics)
            print(f'Lyrics saved for {track_artist} - {track_name}')
        else:
            print(f'No lyrics found for {track_artist} - {track_name}')
    except Exception as e:
        print(f'Failed to fetch lyrics for {track_artist} - {track_name}. Error: {e}')

    # # Fetch additional metadata from Spotify
    # try:
    #     results = sp.search(q=f'artist:{track_artist} track:{track_name}', type='track', limit=1)
    #     if results['tracks']['items']:
    #         track_id = results['tracks']['items'][0]['id']
    #         audio_features = sp.audio_features([track_id])[0]
    #         metadata = {
    #             'Danceability': audio_features['danceability'],
    #             'Energy': audio_features['energy'],
    #             'Key': audio_features['key'],
    #             'Loudness': audio_features['loudness'],
    #             'Mode': audio_features['mode'],
    #             'Speechiness': audio_features['speechiness'],
    #             'Acousticness': audio_features['acousticness'],
    #             'Instrumentalness': audio_features['instrumentalness'],
    #             'Liveness': audio_features['liveness'],
    #             'Valence': audio_features['valence'],
    #             'Tempo': audio_features['tempo']
    #         }
    #         with open(os.path.join(folder_path, 'metadata.txt'), 'w', encoding='utf-8') as f:
    #             for key, value in metadata.items():
    #                 f.write(f'{key}: {value}\n')
    #         print(f'Metadata saved for {track_artist} - {track_name}')
    #     else:
    #         print(f'No Spotify data found for {track_artist} - {track_name}')
    # except Exception as e:
    #     print(f'Failed to fetch Spotify data for {track_artist} - {track_name}. Error: {e}')

    # Pause between downloads to avoid overwhelming the server
    import time
    time.sleep(2)

print('Data collection complete.')