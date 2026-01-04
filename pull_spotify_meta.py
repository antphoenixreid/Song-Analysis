# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 19:01:58 2023

@author: mythk
"""
import csv

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Initial Variables
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="638be10d0e7f44a091b815f77ed280e6",
                                               client_secret="ff842693b14d445cae25161a8ca77972",
                                               redirect_uri="http://127.0.0.1:9090",
                                               scope="user-library-read"))

header = ['Artist', 'Track Title', 'Duration (minutes)', 'Key', 'Time Signature',
          'Tempo', 'Energy', 'Danceability', 'Speechiness', 'Acousticness',
          'Instrumentalness', 'Liveness', 'Valence']

pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

output_path = 'D:/Engineering/Signal Processing/Personal Projects/Song Analysis/dataset/'
csv_file = output_path + 'SpotifyPlaylist_MetaData.csv'

# Read all the Liked Songs on Spotify
count = 0
tracks = {}
for i in range(int(1650/50)):
    results = sp.current_user_saved_tracks(limit=50,offset=count*50)
    for idx, item in enumerate(results['items']):
        temp_track = {}
        track = item['track']
        # print(count*50+idx+1, track['artists'][0]['name'], " – ", track['name'])
        temp_track['Artist'] = track['artists'][0]['name']
        
        # track_artist = sp.artist(track["artists"][0]["external_urls"]["spotify"])
        # temp_track['Genres'] = track_artist["genres"]
        
        temp_track['Track Title'] = track['name']
        temp_track['Duration (minutes)'] = track['duration_ms']/60000

        track_id = track['track']['id']
        track_meta = sp.audio_features(track_id)[0]
        
        temp_track['Key'] = pitch_class[track_meta['key']]
        temp_track['Time Signature'] = track_meta['time_signature']
        temp_track['Tempo'] = track_meta['tempo']
        temp_track['Energy'] = track_meta['energy']
        temp_track['Danceability'] = track_meta['danceability']
        temp_track['Speechiness'] = track_meta['speechiness']
        temp_track['Acousticness'] = track_meta['acousticness']
        temp_track['Instrumentalness'] = track_meta['instrumentalness']
        temp_track['Liveness'] = track_meta['liveness']
        temp_track['Valence'] = track_meta['valence']
        tracks[count*50+idx+1] = temp_track
    count += 1
    
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        for data in tracks:
            try:
                writer.writerow(tracks[data])
            except UnicodeEncodeError:
                continue
except IOError:
    print("I/O error")