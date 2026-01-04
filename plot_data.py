# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 07:28:17 2023

@author: mythk
"""
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

import seaborn as sns

# CSV File
spotify_csv = 'E:/Engineering/Signal Processing/Personal Projects/Song Analysis/dataset/SpotifyPlaylist_MetaData.csv'

# Read the Spotify Data
spotify_data = pd.read_csv(spotify_csv, index_col="Artist", encoding = "ISO-8859-1")

# Plot Histogram (Key)
# sns.histplot(spotify_data['Key'])

# Plot Scatterplots to gauge the correlation to Danceability

# Danceability vs Tempo
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Tempo'], ax=axs)
plt.show()

# Danceability vs Energy
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Energy'], ax=axs)
plt.show()

# Danceability vs Liveness
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Liveness'], ax=axs)
plt.show()

# Danceability vs Instrumentalness
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Instrumentalness'], ax=axs)
plt.show()

# Danceability vs Speechiness
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Speechiness'], ax=axs)
plt.show()

# Danceability vs Acousticness
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Acousticness'], ax=axs)
plt.show()

# Danceability vs Valence
fig, axs = plt.subplots()
sns.regplot(x=spotify_data['Danceability'], y=spotify_data['Valence'], ax=axs)
plt.show()