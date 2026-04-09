from pydub import AudioSegment
import numpy as np

# Create silence 1s
audio = AudioSegment.silent(duration=1000)
audio = audio.set_sample_width(1)
print(audio.sample_width)

try:
    audio2 = audio.set_sample_width(2)
    print(audio2.sample_width)
except Exception as e:
    print(e)
