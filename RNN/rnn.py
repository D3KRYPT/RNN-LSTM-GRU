import numpy as np
import tensorflow as tf
f = open("the-verdict.txt", "r", encoding="utf-8")
text = f.read()
vocab = sorted(set(text))

char2id = {u:i for i,u in enumerate(vocab)}
id2char = {i:u for i,u in enumerate(vocab)}
seq_length = 100
encoded_text = np.array([char2id[i] for i in text])
charDataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = charDataset.batch(seq_length+1, drop_remainder=True)
print("sequences")
for seq in sequences.take(5):
  print(seq.numpy())
def split_into_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_into_input_target) #(input, target pair)
for input_eg, target_eg in dataset.take(1):
   print("input", ''.join(id2char[i.numpy()] for i in input_eg))
   print("output", ''.join(id2char[i.numpy()] for i in target_eg))

BATCH_SIZE = 64
BUFFER = 10000

dataset = (dataset
           .shuffle(BUFFER)
           .batch(BATCH_SIZE, drop_remainder=True))
vocab_size = len(vocab)
embedding_dim = 64
rnn_units = 128
# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = "adam", loss = loss_fn)

history = model.fit(dataset, epochs = 300)

def generate_text(model, start_string, num_char=100):
   input_eval = [char2id[i] for i in start_string]
   input_eval = tf.expand_dims(input_eval, 0)

   text_generated =[]

   for _ in range(num_char):
      predictions = model(input_eval)
      predictions = tf.sq
   
