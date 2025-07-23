import librosa

def convertAudioToMel(audioFilePath):
    y, sr = librosa.load(audioFilePath, offset=30, duration=1)
    s_mel = librosa.feature.melspectrogram(y=y, sr=sr)