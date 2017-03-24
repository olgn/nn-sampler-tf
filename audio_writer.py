import librosa
import numpy as np
import matplotlib.pyplot as plt


def write_wav(waveform, sample_rate, filename):
    librosa.output.write_wav(filename, waveform, sample_rate)
    print('Generated wav file at {}'.format(filename))

def plot_wav(waveform):
    plt.plot(waveform)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.title('Generated Audio')
    plt.show()

def generate_noise(samples):
    return np.transpose(np.reshape(np.random.uniform(-1,1,samples),[-1,1]))

def write(network, sample_rate, num_samples, window_size, filename):
    waveform = np.array([])
    seed = generate_noise(window_size)
    input = seed
    for _ in range(num_samples):
        # print('input shape:', np.shape(input[0][-window_size:]))
        y = network.predict([input[0][-window_size:]])
        waveform = np.append(waveform, y)
        input = np.append(input, y, 1)

    print('shape of output waveform:', np.shape(waveform))
    # write_wav(waveform, sample_rate, filename)
    plot_wav(waveform)


