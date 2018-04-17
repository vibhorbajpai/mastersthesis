import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load sound file
y, sr = librosa.load("/Users/Vibhor/Documents/AcademicsAndCV/UPF/MIR/newmodels/datasets/genre_rosamerica/audio/mp3/roc/creed-my_sacrifice.mp3")

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.stft(y)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a linear scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='linear')

# Put a descriptive title on the plot
plt.title('linear spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()
plt.show()