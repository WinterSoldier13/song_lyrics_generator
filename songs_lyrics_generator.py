import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical, normalize
from random import randint

# READING THE DATA
file = open('D://training_data/lyrics.txt', 'r')
text = file.read()
file.close()


chars = sorted(list(set(text)))          # Contains all the characters
# create mapping of characters to integers (0-25) and the reverse
char2int = dict((c, i) for i, c in enumerate(chars))
int2char = dict((i, c) for i, c in enumerate(chars))

print(len(chars))

step = 5

sequence_length = 40

input_seq = []
input_seq1 = []
output_seq = []
output_seq1 = []

for i in range(0, l - sequence_length, step):
    input_ = text[i:i + sequence_length]
    output_ = text[i + sequence_length]
    input_seq1.append(input_)
    output_seq1.append(output_)
    input_seq.append([char2int[ch] for ch in input_])
    output_seq.append([char2int[ch] for ch in output_])



# ONE-HOT ENCODING
x_data = np.zeros((len(input_seq),sequence_length,len(chars)))
# normalize
for i, sentence in enumerate(input_seq1):
    for t, char in enumerate(sentence):
        x_data[i, t, char2int[char]] = 1

# one hot encode the output variable
y_data = to_categorical(output_seq)



print(np.shape(x_data))
print(np.shape(y_data))





#####################################################################################################
model = Sequential()
model.add(LSTM(256,input_shape=(40,42),return_sequences=True))
model.add(Dropout(0.8))
model.add(LSTM(128))
model.add(Dropout(0.8))
model.add(Dense(42,activation='softmax'))
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_data,y_data, validation_split=0.2, batch_size=200, epochs=300)

######################################################################################################



#########
# SAVING THE MODEL
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#########

''' LOADING THE SAVED MODEL
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''


#######################################################################################################

# Making Predictions

#######################################################################################################


def onehot2num(arr):
    c = 0
    index = 0
    flag = True

    for i in y_data:

        i = list(i)

        if i == list(arr):
            # print('Found at:')
            # print(c)
            index = c
            flag = False
            break
        c += 1
    if flag:
        print('NOT FOUND')

    return output_seq[index]




x = x_data[1:2]
start_index = randint(1, len(x_data))
# x = x_data[start_index:start_index+1]
x_chars = list(input_seq1[1:2])
# x_chars = list(input_seq1[start_index:start_index+1])
outy = x_chars

for i in range(1000):

    predictions = model.predict(x)
    prediction = predictions[0]

    alpha = np.zeros((1, 42))
    # print(np.argmax(prediction))
    alpha[0][np.argmax([prediction])] = 1

    beta = onehot2num(alpha[0])

    ch = str(int2char[beta[0]])
    outy.append(ch)
    x_chars = [(x_chars[0] + ch)[1:]]

    x = np.zeros((1, sequence_length, len(chars)))
    # normalize
    for i, sentence in enumerate(x_chars):
        for t, char in enumerate(sentence):
            x[i, t, char2int[char]] = 1
            # print(i,sentence)

    predictions = [[]]
    prediction = []

output_sentence = ''
for word in outy:
    output_sentence += word
#########




print(output_sentence)