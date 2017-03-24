import tensorflow as tf
import numpy as np
import audio_reader as reader
import audio_writer as writer
import network as net


#set global variables:
FILE_DIRECTORY = 'C://Users/Teal/Documents/Python/Projects/Machine Learning/nn-sampler/scripts/kicks' #where those files @?
SAMPLE_RATE = 44100 #the sample rate at which we read / write audio data
WINDOW_SIZE = 1024 #how many data points do we present as input?
NUM_STEPS = 10 #how many times we're gonna loop through the sample folder
GENERATE_LENGTH = 44100 #how many data points of audio will be generated in our output files?
GENERATED_FILE_PATH = 'C://Users/Teal/Documents/Python/Projects/Machine Learning/nn-sampler-tf/outputs/kicks/'
ITERATIONS_BEFORE_GENERATION = 50000 #number of training steps before we generate an audio file
ITERATIONS_BEFORE_PRINT_LOSS = 500 #number of training steps before we print the net loss

LAYER_SIZES = [WINDOW_SIZE, 2048, 1024, 256, 64, 8, 1] # layers, starting with the input layer, ending with output
ACTIVATION_FUNCTION = tf.tanh #what activation function we gonna use?

#setup the nn:
nn = net.Network(LAYER_SIZES, ACTIVATION_FUNCTION)

#loop through the training stage
sample = 1
training_iterations = 1
file_number = 0
for step in range(NUM_STEPS):
    # get the audio data generator:
    audio_file_iterator = reader.load_generic_audio(FILE_DIRECTORY, SAMPLE_RATE)

    #loop through all files in the audio folder,
    for audio, filename in audio_file_iterator:
        print('filename:', filename)
        print('shape of audio', np.shape(audio))
        buffer_ = np.array([])

        buffer_ = np.append(buffer_, audio)
        left = 0
        right = WINDOW_SIZE
        while left < len(buffer_) - WINDOW_SIZE - 1:
            #slice a piece of the buffer for sampling
            piece = np.transpose(np.reshape(buffer_[left:right], [-1,1]))
            target = np.array([buffer_[right + 1]])
            #train on this sample, using the next piece as a target
            nn.train(piece, target)

            #print the loss every 50 steps:
            if left % ITERATIONS_BEFORE_PRINT_LOSS == 0:
                print('step:', step, '; sample:', sample, '; iteration:', training_iterations - 1)
                nn.print_loss(piece, target)
                predicted = nn.predict(piece)
                print('actual output:', target, 'predicted output:', predicted)

            if training_iterations % ITERATIONS_BEFORE_GENERATION == 0:
                print('generating audio file...')
                filename = GENERATED_FILE_PATH + file_number.__str__() + '.wav'
                writer.write(nn, SAMPLE_RATE, GENERATE_LENGTH, WINDOW_SIZE, filename)
                file_number +=1
            # shift the window
            left += 1
            right = left + WINDOW_SIZE
            training_iterations +=1

        sample += 1