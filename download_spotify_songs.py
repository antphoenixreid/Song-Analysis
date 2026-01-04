import spotipy
from spotipy.oauth2 import SpotifyOAuth

import os

from pytube import YouTube, Search
import numpy as np

def is_substring_in_string(substring, full_string):
    # Split both strings into sets of words (ignoring case)
    substring_set = set(substring.lower().split())
    full_string_set = set(full_string.lower().split())

    # Check if the substring set is a subset of the full string set
    return substring_set.issubset(full_string_set)

# Set up Spotify API Access
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="638be10d0e7f44a091b815f77ed280e6",
                                               client_secret="ff842693b14d445cae25161a8ca77972",
                                               redirect_uri="http://127.0.0.1:9090",
                                               scope="user-library-read"))

# Read all the Liked Songs on Spotify
count = 0
tracks = []
skippable_kw = ['Eminem Public Service Announcement', 'Eminem Public Service Announcement 2000', 'Que Rock Hardships']

passable_tracks = ['Kirishima Rap (Red Riot)', 'Wolverine vs Freddy Krueger', 'Let Him Cook (Sanji)']

for i in range(int(1500/50)):
    try:
        results = sp.current_user_saved_tracks(limit=50,offset=count*50)
    except:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="638be10d0e7f44a091b815f77ed280e6",
                                                       client_secret="ff842693b14d445cae25161a8ca77972",
                                                       redirect_uri="http://127.0.0.1:9090",
                                                       scope="user-library-read"))
        results = sp.current_user_saved_tracks(limit=50,offset=count*50)
        
    for idx, item in enumerate(results['items']):
        temp_track = {}
        track = item['track']
        track_artist = track['artists'][0]['name']
        track_name = track['name']
        track_duration = track['duration_ms']/60000
        # print(count*50+idx+1, track['artists'][0]['name'], " – ", track['name'])
        # temp_track['Artist'] = track['artists'][0]['name']
        # temp_track['Track Title'] = track['name']
        # temp_track['Duration (minutes)'] = track['duration_ms']/60000
        # tracks[count*50+idx+1] = temp_track
        
        # Search for the video on YouTube
        print('Preparing ' + track_artist + '-' + track_name)
        search_kw = track_artist + ' ' + track_name
        
        if search_kw in tracks or search_kw in skippable_kw:
            break
        
        tracks.append(search_kw)
        s = Search(search_kw)
        
        # Parse through search results for optimal video
        for k in range(len(s.results)):
            try:
                len_dif = 999
                video_id = s.results[k].video_id
                temp_song_yt_link = "https://www.youtube.com/watch?v=" + video_id
                yt_track_length = YouTube(temp_song_yt_link).length/60
                yt_track_title = YouTube(temp_song_yt_link).streams[0].title
                
                # Check if title is the same as spotify song (and not an instrumental)
                # if is_substring_in_string(track_name, yt_track_title) or is_substring_in_string(track_artist, yt_track_title) and yt_track_title.find('Instrumental') == -1:
                #     if np.abs(track_duration - yt_track_length) < len_dif:
                #         len_dif = np.abs(track_duration - yt_track_length)
                #         # If the song is close to the length of the Spotify song, break loop and save YT link
                #         if len_dif < 1e-1:
                #             song_yt_link = temp_song_yt_link
                #             print('Found ' + track_artist + '-' + track_name)
                #             break
                #         elif track_name in passable_tracks:
                #             song_yt_link = temp_song_yt_link
                #             print('Found ' + track_artist + '-' + track_name)
                #             break
                # elif track_name.find(yt_track_title) != -1 and yt_track_title.find('Instrumental') == -1:
                #     if np.abs(track_duration - yt_track_length) < len_dif:
                #         len_dif = np.abs(track_duration - yt_track_length)
                #         # If the song is close to the length of the Spotify song, break loop and save YT link
                #         if len_dif < 2e-1:
                #             song_yt_link = temp_song_yt_link
                #             print('Found ' + track_artist + '-' + track_name)
                #             break
                # else:
                #     if np.abs(track_duration - yt_track_length) < len_dif:
                #         len_dif = np.abs(track_duration - yt_track_length)
                #         # If the song is close to the length of the Spotify song, break loop and save YT link
                #         if len_dif < 2e-1:
                #             song_yt_link = temp_song_yt_link
                #             print('Found ' + track_artist + '-' + track_name)
                #             break
                        
                if yt_track_title.find(track_name) != -1 or yt_track_title.find(track_artist) != -1 and yt_track_title.find('Instrumental') == -1:
                    if np.abs(track_duration - yt_track_length) < len_dif:
                        len_dif = np.abs(track_duration - yt_track_length)
                        # If the song is close to the length of the Spotify song, break loop and save YT link
                        if len_dif < 2e-1 or track_name == 'Kirishima Rap (Red Riot)':
                            song_yt_link = temp_song_yt_link
                            print('Found ' + track_artist + '-' + track_name)
                            break
                
                elif track_name.find(yt_track_title) != -1 and yt_track_title.find('Instrumental') == -1:
                    if np.abs(track_duration - yt_track_length) < len_dif:
                        len_dif = np.abs(track_duration - yt_track_length)
                        # If the song is close to the length of the Spotify song, break loop and save YT link
                        if len_dif < 2e-1:
                            song_yt_link = temp_song_yt_link
                            print('Found ' + track_artist + '-' + track_name)
                            break
                elif yt_track_title == track_name:
                    song_yt_link = temp_song_yt_link
                    print('Found ' + track_artist + '-' + track_name)
                    break
                elif is_substring_in_string(track_name, yt_track_title) or is_substring_in_string(track_artist, yt_track_title) and yt_track_title.find('Instrumental') == -1:
                    if np.abs(track_duration - yt_track_length) < len_dif:
                        len_dif = np.abs(track_duration - yt_track_length)
                        # If the song is close to the length of the Spotify song, break loop and save YT link
                        if len_dif < 2e-1:
                            song_yt_link = temp_song_yt_link
                            print('Found ' + track_artist + '-' + track_name)
                            break
                        elif track_name in passable_tracks:
                            song_yt_link = temp_song_yt_link
                            print('Found ' + track_artist + '-' + track_name)
                            break
                else:
                    if np.abs(track_duration - yt_track_length) < len_dif:
                        len_dif = np.abs(track_duration - yt_track_length)
                        # If the song is close to the length of the Spotify song, break loop and save YT link
                        if len_dif < 2e-1:
                            song_yt_link = temp_song_yt_link
                            print('Found ' + track_artist + '-' + track_name)
                            break
            except:
                continue
            
        # Extract the audio to download
        selected_video = YouTube(song_yt_link).streams.filter(only_audio=True).first()
        output_path = 'D:/Engineering/Signal Processing/Personal Projects/Song Analysis/dataset/songs/'
        
        # If title does not match the spotify title, do not download (just in case)
        # if audio_title.find(track_name) != -1 or audio_title.find(track_artist) != -1:
        
        print('Downloading ' + track_artist + '-' + track_name)
        
        try:    
            audio = selected_video.download(output_path=output_path)
            audio_title = audio.title()
            
            base, ext = os.path.splitext(audio)
            new_file = base + '.mp3'
            os.rename(audio, new_file)
            
            print('Download of ' + track_artist + '-' + track_name + ' complete!')
        except:
            print('Issue detected! Deleting ' + track_artist + '-' + track_name)
            
            if os.path.exists(audio):
                os.remove(audio)
    
    count += 1