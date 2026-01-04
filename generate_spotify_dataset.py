import spotipy
from spotipy.oauth2 import SpotifyOAuth

import urllib.request
from unidecode import unidecode
import re, os

from pytube import YouTube, Search
# from tube_dl import Youtube, extras

from moviepy.editor import *
import numpy as np

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="1d96de8cf09f4ff89a0189d2c864e6c0",
                                               client_secret="a0b53ad8827b4505ac85a93cf7d7f523",
                                               redirect_uri="http://127.0.0.1:9090",
                                               scope="user-library-read"))

# Read all the Liked Songs on Spotify
count = 0
tracks = {}
for i in range(int(1500/50)):
    results = sp.current_user_saved_tracks(limit=50,offset=count*50)
    for idx, item in enumerate(results['items']):
        temp_track = {}
        track = item['track']
        # print(count*50+idx+1, track['artists'][0]['name'], " – ", track['name'])
        temp_track['Artist'] = track['artists'][0]['name']
        temp_track['Track Title'] = track['name']
        temp_track['Duration (minutes)'] = track['duration_ms']/60000
        tracks[count*50+idx+1] = temp_track
    count += 1

# Download Audio from YouTube
error_data = []
for i in range(len(tracks)):
    try:
        search_kw = tracks[i+1]['Artist'] + ' ' + tracks[i+1]['Track Title']

        print('Preparing ' + tracks[i+1]['Artist'] + '-' + tracks[i+1]['Track Title'])
        
        spot_track_length = tracks[i+1]['Duration (minutes)']

        # search_kw = search_kw.replace(' ','+')

        # html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + unidecode(search_kw))
        # video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        # video_ids = list(set(video_ids))
        # video_ids = video_ids[:20]
        
        s = Search(search_kw)
        
        for k in range(len(s.results)):
            len_dif = 999
            video_id = s.results[k].video_id
            
            try:
                temp_song_yt_link = "https://www.youtube.com/watch?v=" + video_id
                yt_track_length = YouTube(temp_song_yt_link).length/60
                yt_track_title = YouTube(temp_song_yt_link).streams[0].title
                print(yt_track_title + ' ' + str(yt_track_length))
            except:
                continue
            
            if yt_track_title.find(tracks[i+1]['Track Title']) != -1 and yt_track_title.find('Instrumental') == -1:
                if np.abs(spot_track_length - yt_track_length) < len_dif:
                    len_dif = np.abs(spot_track_length - yt_track_length)
                    song_yt_link = temp_song_yt_link
                    
        # song_yt_link = "https://www.youtube.com/watch?v=" + video_id
        # yt_track_length = YouTube(song_yt_link).length/60
        # yt_track_title = YouTube(song_yt_link).streams[0].title
                
        selected_video = YouTube(song_yt_link).streams.filter(only_audio=True).first()

        output_path = 'E:/Engineering/Signal Processing/Personal Projects/Song Analysis/dataset/songs/'
        # audio = selected_video.streams.filter(file_extension='mp4').first()
        # audio.download(output_path=output_path)
        
        print('Downloading ' + tracks[i+1]['Artist'] + '-' + tracks[i+1]['Track Title'])
        audio = selected_video.download(output_path=output_path)
        # extras.Convert(audio,'mp3',add_meta=True)

        title = audio.title
        # title = title.replace("'","")
        # title = title.replace(":","")
        # title = title.replace(";","")
        # title = title.replace(",","")
        # title = title.replace("#","")
        # # title = title.replace("&","")
        # title = title.replace("{","")
        # title = title.replace("}","")
        # title = title.replace("<","")
        # title = title.replace(">","")
        # title = title.replace("*","")
        # title = title.replace("?","")
        # title = title.replace("!","")
        # title = title.replace("@","")
        # title = title.replace("+","")
        # title = title.replace("=","")
        # title = title.replace("|","")
        # title = title.replace("$","")
        # title = title.replace(".","")
        
        print('Saving ' + tracks[i+1]['Artist'] + '-' + tracks[i+1]['Track Title'])
        base, ext = os.path.splitext(audio)
        new_file = base + '.mp3'
        os.rename(audio, new_file)
        
        print('Saved ' + tracks[i+1]['Artist'] + '-' + tracks[i+1]['Track Title'])
    except:
        error_data.append(title)
        continue
