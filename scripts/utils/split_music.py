from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import listdir
from os.path import isfile, join
import os

def split_song(filename, song, interval = 13 * 1000):
    for i in xrange(0, len(song), interval):
        piece = song[i:i+interval]
        index = i / interval + 1
        piece.export(filename + "_" + str(index) + ".wav", format="wav")

def split(audio_file):
    filename, file_extension = os.path.splitext(audio_file)
    if file_extension == ".mp3":
        song = AudioSegment.from_mp3(audio_file)
        split_song(filename, song)
    elif file_extension == ".wav":
        song = AudioSegment.from_wav(audio_file)
        split_song(filename, song)

def main():
    mypath = "/Users/dongshu/courses/statistics/music/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for audio_file in onlyfiles:
        split(mypath + audio_file)

if __name__ == "__main__":
    main()

