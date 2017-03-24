import numpy as np
import audio_reader as reader

def get_data(path):
    SAMPLE_RATE = 44100
    WINDOW_SIZE = 512
    audio_file_iterator = reader.load_generic_audio(path, SAMPLE_RATE)
    for audio, filename in audio_file_iterator:


