#!/usr/bin/env python
from __future__ import print_function

import csv
import librosa
import sys
import argparse
import numpy
import pandas
from os import listdir
from os.path import isfile, join

def completeTrain(name_csv, path):

    file1 = open(name_csv, "r")
    trainFinal = open('data/trainComplete.csv', "wb")
    writer = csv.writer(trainFinal)
    writer.writerow(('songname', 'genre', 'tempo', 'beats', 'chromagram_stft', 'chromagram_cqt',
    'rmse', 'spectral_cent', 'spectral_bw', 'spectral_rolloff', 'zero_crossing', 'mfcc1', 'mfcc2',
    'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
    'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'))

    file_data = [f for f in sorted(listdir(path)) if isfile (join(path, f))]
    reader = csv.reader(file1)
    next(reader)

    for line in file_data:
        if ( line[-1:] == '\n' ):
            line = line[:-1]

        features = []
        songloc = path + '/' + line
        songname = line
        songname = songname[-10:-4]
        i = 5
        while songname[0]=='0':
            songname = songname[-i:]
            i = i - 1

        print()
        print("Reading Song: "+ songname +".mp3")

        y, sr = librosa.load(songloc, duration=30)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        row = reader.next()
        name = row[0]
        genre = row[1]
        if (songname == name):
            genre = genre - 1
            songgenre = genre

        beatsWR = numpy.sum(beats)
        chroma_stftWR = numpy.mean(chroma_stft)
        chroma_cqtWR = numpy.mean(chroma_cqt)
        rmseWR = numpy.mean(rmse)
        spectral_centWR = numpy.mean(spectral_cent)
        spectral_bwWR = numpy.mean(spectral_bw)
        spectral_rolloffWR = numpy.mean(spectral_rolloff)
        zcrWR = numpy.mean(zcr)
        for coefficient in mfcc:
            features.append(numpy.mean(coefficient))

        writer.writerow((songname,songgenre,tempo,beatsWR,chroma_stftWR, chroma_cqtWR,
        rmseWR,spectral_centWR, spectral_bwWR, spectral_rolloffWR, zcrWR, features[0],features[1],features[2],features[3],
        features[4],features[5],features[6],features[7],features[8],features[9],
        features[10],features[11],features[12],features[13],features[14],features[15],
        features[16],features[17],features[18],features[19]))

    file1.close()
    trainFinal.close()

# main
completeTrain('data/train.csv','train/Train')
