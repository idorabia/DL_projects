import math
import matplotlib.pyplot as plt
import pretty_midi as pm
import numpy as np
import csv
import os
import gensim
import datetime
import multiprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Embedding, Input, concatenate
from scipy.spatial.distance import cosine
from gensim.models import FastText

MIDI_PATH = './midi_files'
SEED = 42
np.random.seed(SEED)


def rename_files_in_folder(folder):
    """
    Renaming all files in folder to lower-case
    """
    for f in os.listdir(folder):
        os.rename(f"{folder}/{f}", f"{folder}/{f.lower()}")


def load_midi_file(midi_path, artist, song):
    """
    Loading midi files into PrettyMIDI object.
    Returns None if midi file is corrupted.
    """
    artist_with_underscores = artist.replace(' ', '_')
    song_with_underscores = song.replace(' ', '_')
    try:
        midi = pm.PrettyMIDI(f"{midi_path}/{artist_with_underscores}_-_{song_with_underscores}.mid")
        midi.remove_invalid_notes()
        return midi
    except Exception as e:
        return None


def get_midi_data(lyrics_data):
    """
    This function parse the midi file of each song in lyrics_data.
    For each song it returns a dictionary with the song's end_time, and the instrumental data (participating instruments and their played notes).
    """
    midi_data = {}
    for song in lyrics_data:
        artist = song['artist']
        song_name = song['song_name'].lstrip()  # Removing leading whitespaces (songs in test set are with whitespaces)

        #  Loading relevant midi file and get its embedding
        midi_file = load_midi_file(MIDI_PATH, artist, song_name)
        artist_with_underscores = artist.replace(' ', '_')
        song_with_underscores = song_name.replace(' ', '_')

        #  Storing relevant midi information (end_time and instruments list)
        song_relevant_midi_data = []
        if midi_file == None:
            midi_data[f"{artist_with_underscores}_-_{song_with_underscores}"] = None
        else:
            for instrument in midi_file.instruments:
                program = int(instrument.program)
                notes = instrument.notes
                song_relevant_midi_data.append([program, notes])
            midi_data[f"{artist_with_underscores}_-_{song_with_underscores}"] = {'end_time': midi_file.get_end_time(), 'instruments': song_relevant_midi_data}

    return midi_data


def clean_text(text):
    """
    Clean the text from special characters.
    Inserts EOS and EOL tokens.
    """
    text = text.replace('&', 'EOL')  # End-Of-Line token
    text += ' EOS'  # End-Of-Sentence token
    return np.array(gensim.utils.simple_preprocess(text))


def load_lyrics(lyrics_path):
    """
    Loading lyrics file from csv
    """
    with open(lyrics_path, encoding='utf8') as spreadshit:
        lyrics_train_set = csv.reader(spreadshit)
        data = [{'artist': sample[0],
                       'song_name': sample[1],
                       'song_text': sample[2],
                       'clean_song_text': clean_text(sample[2])}
                      for sample in lyrics_train_set]

    #  Dropping duplicates
    import collections
    songs_list = [f"{song['artist']}_-_{song['song_name']}" for song in data]
    duplicated_songs = [song for song, count in collections.Counter(songs_list).items() if count > 1]
    duplicated_songs_indices = [songs_list.index(song) for song in duplicated_songs]

    for duplicated_song_index in duplicated_songs_indices:
        data.pop(duplicated_song_index)

    return data


def get_midi_simple_embedding(midi_object, words):
    """
    Generates embedding for the midi-file according to the number of instruments participating in the entire song.
    Returns vector of zeros if midi_file is None (corrupted)
    """
    total_words = len(words)
    midi_embeddings = np.zeros((total_words, 128), dtype=np.int32)
    if midi_object == None:
        return midi_embeddings
    else:
        song_instruments = midi_object['instruments']
        active_instruments_indices = [instrument[0] for instrument in song_instruments]

        instruments_vector = np.zeros(128, dtype=np.int32)
        instruments_vector[active_instruments_indices] = 1
        midi_embeddings = np.tile(instruments_vector, total_words).reshape(total_words, 128)
        return midi_embeddings


def _find_first_note_in_time(word_time, notes, middle=None, index_tracks = None):
    """
    Search the first and the last notes that have been played between (word_time - 1) and (word_time + 1).
    We use here binary-search to save time.
    """
    total_notes = len(notes)

    #  Define index_tracks as a list of all visited indices (to prevent eternal loop)
    if index_tracks == None:
        index_tracks = []
    if middle == None:
        middle = int(math.floor(total_notes / 2))

    curr_note = notes[middle]
    #  Edge cases for words without melody
    if notes[0].start - 1 > word_time or notes[-1].end + 1 < word_time:
        return None, None
    elif middle in index_tracks:
        return None, None

    #  Searching recursively
    elif curr_note.start - 1 > word_time:
        new_middle = middle - int(math.ceil(middle/2))
        index_tracks.append(middle)
        return _find_first_note_in_time(word_time, notes, new_middle, index_tracks)
    elif curr_note.end + 1 < word_time:
        new_middle = middle + int(math.floor((total_notes-middle)/2))
        index_tracks.append(middle)
        return _find_first_note_in_time(word_time, notes, new_middle, index_tracks)

    #  Find the longest sequence of notes played during (word_time - 1) and (word_time + 1)
    else:
        first_note = middle
        last_note = middle

        curr_note = notes[first_note]
        if first_note > 0:
            while curr_note.start - 1 <= word_time <= curr_note.end + 1 and first_note > 0:
                first_note -= 1
                curr_note = notes[first_note]
            first_note += 1

        curr_note = notes[last_note]
        if last_note < total_notes - 1:
            while curr_note.start - 1 <= word_time <= curr_note.end + 1 and last_note < total_notes -1:
                last_note += 1
                curr_note = notes[last_note]
            last_note -= 1
        return first_note, last_note


def get_midi_complex_embedding(midi_object, words):
    """
    Generates embedding for the midi-file according to the number of instruments participating during the approximated time of the word.
    In addition, it includes the notes played during the approximated time of the word.
    Returns vector of zeros if midi_file is None (corrupted)
    """
    total_words = len(words)
    midi_embeddings = np.zeros((total_words, 256), dtype=np.int32)
    if midi_object == None:
        return midi_embeddings

    song_in_seconds = midi_object['end_time']
    song_instruments = midi_object['instruments']

    for i in range(total_words):
        word_relative_time = (i/total_words) * song_in_seconds
        instruments_vector = np.zeros(128, dtype=np.int32)
        notes_vector = np.zeros(128, dtype=np.int32)

        for instrument in song_instruments:
            instrument_number = instrument[0]
            notes = instrument[1]

            #  Finding the first note and the last note for the current word.
            first_note_for_word, last_note_for_word = _find_first_note_in_time(word_relative_time, notes)
            if first_note_for_word == None:
                pass
            else:
                for note_index in range(first_note_for_word, last_note_for_word+1):
                    instruments_vector[instrument_number] = 1  # instruments participated in this word_time
                    notes_vector[notes[note_index].pitch] = 1  # notes played during this word_time
        current_word_melody = np.concatenate([instruments_vector, notes_vector])
        midi_embeddings[i] = current_word_melody

    return midi_embeddings


def train_word2vec(data):
    """
    Train a FastText Word2Vec model on our training-songs corpus
    """
    cpu_amt = multiprocessing.cpu_count()
    clean_songs = [list(song_record['clean_song_text']) for song_record in data]
    word2vec_model = FastText(sentences=clean_songs, size=300, workers=cpu_amt, min_count=1, seed=SEED, iter=200)
    return word2vec_model


def preprocess_test_data(embedding_model, data, midi_data, midi_embedding_func):
    """
    Preprocess test-data in the following manner:
    1. First, it tokenizes the song's words using the embedding_model (each word receives its index in the embedding_model).
    2. Next, it prepares the melody_embedding using the given midi_embedding_func.
    3. Last, it returns X as list of size <number_of_songs>.
       Each entry is a pair of the first word in the song, and the melody_embedding for all the words.
    """

    X = []
    for song in data:
        artist = song['artist']
        song_name = song['song_name'].lstrip()
        song_words = song['clean_song_text']

        #  Loading relevant midi file and get its embedding
        artist_with_underscores = artist.replace(' ', '_')
        song_with_underscores = song_name.replace(' ', '_')
        midi_key = f"{artist_with_underscores}_-_{song_with_underscores}"
        midi_object = midi_data[midi_key]
        midi_file_embeddings = midi_embedding_func(midi_object, song_words)

        #  Defining the first word of the song:
        X.append(['love', midi_file_embeddings, midi_key])
        X.append(['cold', midi_file_embeddings, midi_key])
        X.append(['beer', midi_file_embeddings, midi_key])

    return X


def preprocess_train_data(embedding_model, data, midi_data, midi_embedding_func):
    """
    Preprocess train-data in the following manner:
    1. First, it tokenizes the song's words using the embedding_model (each word receives its index in the embedding_model).
    2. Next, it prepares the melody_embedding using the given midi_embedding_func.
    3. Last, it returns X as list of size <number_of_songs>.
       Each entry is a list of the song's words and it's melody_embeddings.
       In addition, it return Y, which is a list of consecutive words for X.
    """
    def get_tokens(word):
        return embedding_model.wv.vocab[word].index

    X = []
    Y = []
    for song in data:
        artist = song['artist']
        song_name = song['song_name']
        song_tokens = np.vectorize(get_tokens)(song['clean_song_text'])

        #  Loading relevant midi object and get its embedding
        artist_with_underscores = artist.replace(' ', '_')
        song_with_underscores = song_name.replace(' ', '_')
        midi_key = f"{artist_with_underscores}_-_{song_with_underscores}"
        midi_object = midi_data[midi_key]
        midi_file_embeddings = midi_embedding_func(midi_object, song_tokens)

        #  Generating pairs of (current_word, next_word)
        curr_song_X_lyrics = song_tokens[:-1]
        curr_song_X_lyrics = curr_song_X_lyrics.reshape(len(curr_song_X_lyrics), 1)
        curr_song_X_midi = midi_file_embeddings[:-1].reshape(len(song_tokens) - 1, 1, -1)
        curr_song_Y = song_tokens[1:].reshape(len(song_tokens) - 1, 1)

        X.append([curr_song_X_lyrics, curr_song_X_midi])
        Y.append(curr_song_Y)
    return X, Y


def input_generator(X, Y):
    """
    Wraps X and Y with a generator.
    This way, each song would be a batch for the LSTM model.
    """
    while True:
        for i in range(len(X)):
            yield X[i], Y[i]


def generate_song(RNN_model, embedding_model, melody_embeddings, initial_word, max_words=1000):
    """
    This function uses RNN_model to create a song, based on an initial word and the melody_embedding.
    It runs the trained RNN_model until it predicts the word 'EOS' (End-of-Song), or until it reaches max_words.
    """
    melody_embedding_size = melody_embeddings.shape[1]
    total_melody_embeddings = len(melody_embeddings)
    melody_embeddings = melody_embeddings.reshape((total_melody_embeddings,1,-1))

    vocabulary = embedding_model.wv.vocab
    vocabulary_indices = [i for i in range(len(vocabulary))]
    predicted_lyrics_indices = [vocabulary[initial_word].index]
    for word_index in range(max_words):
        input_word = np.array(predicted_lyrics_indices[word_index]).reshape((1,1))
        if word_index >= total_melody_embeddings:
            current_melody_embedding = np.zeros((1, 1, melody_embedding_size), dtype=np.int32)
        else:
            current_melody_embedding = melody_embeddings[word_index].reshape(1,1,melody_embedding_size)

        word_prediction_distribution = RNN_model.predict([input_word, current_melody_embedding])[0][0]
        # word_prediction_distribution = model.predict(batch_generator(test_x))
        predicted_word_index = np.random.choice(vocabulary_indices, replace=True, p=word_prediction_distribution)
        predicted_lyrics_indices.append(predicted_word_index)
        if embedding_model.wv.index2word[predicted_word_index] == 'eos':
            break

    predicted_lyrics = [embedding_model.wv.index2word[word_index] for word_index in predicted_lyrics_indices]
    return predicted_lyrics


def create_lstm_model(embedding_model, melody_embedding_features, LSTM_layer_size=200):
    """
    Create and LSTM model that receives a word and a melody embedding, and predict the next word of the song.
    """
    vocabulary_size = len(embedding_model.wv.vocab)
    #vocabulary_size = len(embedding_model.wv.key_to_index)
    embedding_matrix = word2vec_model.wv.vectors
    lyrics_embedding_features = embedding_model.wv.vector_size

    print(f"Vocabulary Size = {vocabulary_size}")
    input_lyrics = Input((1,))
    input_melody = Input((1,melody_embedding_features))

    embedding_lyrics = (Embedding(vocabulary_size, lyrics_embedding_features, weights=[embedding_matrix], input_length=1))(input_lyrics)

    embedding_concat = concatenate([embedding_lyrics, input_melody])

    lstm = (LSTM(LSTM_layer_size, return_sequences=True))(embedding_concat)

    output = Dense(vocabulary_size, activation='softmax')(lstm)

    model = Model([input_lyrics, input_melody], output)
    return model

def get_song_word2vec_similarity(embedding_model, original, generated):
    """
    Evaluates the cosine similarity between the generated song and the original one.
    The evaluation is done by calculating the PAIRWISE cosine similarity between each pair of word-vectors from original and generated.
    """
    original_without_eol = list(filter(lambda a: a != 'eol', original))
    generated_without_eol = list(filter(lambda a: a != 'eol', generated["generated_song"]))
    inspected_len = min(len(original_without_eol), len(generated_without_eol))
    original_vectors = [embedding_model[word] for word in original_without_eol[:inspected_len]]
    generated_vectors = [embedding_model[word] for word in generated_without_eol[:inspected_len]]
    return np.nanmean([cosine(original_vectors[i], generated_vectors[i]) for i in range(inspected_len)])



def get_song_doc2vec_similarity(embedding_model, original, generated):
    """
    Evaluates the cosine similarity between the generated song and the original one.
    The evaluation is done by calculating the OVERALL cosine similarity between the document-vector of the original song,
        and the document-vector of the generated song.
    """
    vectorized_original = embedding_model[' '.join(list(filter(lambda a: a != 'eol', original)))]
    vectorized_generated = embedding_model[' '.join(list(filter(lambda a: a != 'eol', generated['generated_song'])))]
    return cosine(vectorized_original, vectorized_generated)



def evaluate_generated_songs(embedding_model, test_lyrics, generated_songs):
    """
    Evaluated the cosine-similarity between the generated songs and the original songs.
    The evaluation is done both in the document-level and in the word-level, using Word2Vec and Doc2Vec techniques.
    """
    clean_test_songs = [list(song_record['clean_song_text']) for song_record in test_lyrics]
    #
    # # TODO: ADD - spacy.cli.download('en_core_web_sm')
    # en_core_web_sm = spacy.load('en_core_web_sm')  # make sure to use larger package!
    doc2vec_similarity = []
    word2vec_similarity = []

    for i in range(len(clean_test_songs)):
        doc2vec_similarity.append(
            get_song_doc2vec_similarity(embedding_model, clean_test_songs[i], generated_songs[i]))

        word2vec_similarity.append(
            get_song_word2vec_similarity(embedding_model, clean_test_songs[i], generated_songs[i]))

    doc2vec_similarity_mean = np.array(doc2vec_similarity).mean()
    doc2vec_similarity_std = np.array(doc2vec_similarity).std()

    word2vec_similarity_mean = np.array(word2vec_similarity).mean()
    word2vec_similarity_std = np.array(word2vec_similarity).std()
    return word2vec_similarity_mean, word2vec_similarity_std, doc2vec_similarity_mean, doc2vec_similarity_std

def generate_evaluation_figure(simple_mean, simple_std, complex_mean, complex_std, title):
    """
    Generates a bar-graph for the evaluation of the models.
    """
    embedding_techniques = ["simple", "complex"]
    fig_1 = plt.figure()
    fig_1 = plt.bar(embedding_techniques, [simple_mean, complex_mean], width=0.4, yerr=[simple_std, complex_std], ecolor="red", capsize=5)
    fig_1 = plt.ylim([0, 1])
    fig_1 = plt.title(f"{title}: simple VS complex")
    fig_1 = plt.show()





#  General preprocessing
rename_files_in_folder(MIDI_PATH)
train_lyrics = load_lyrics('./lyrics_train_set.csv')
midi_data = get_midi_data(train_lyrics)
word2vec_model = train_word2vec(train_lyrics)
compilation_params = dict(optimizer='adam', loss='sparse_categorical_crossentropy')




################################################
############## Simple Model ####################
################################################
#  Preprocessing the data for the simple_model
X, Y = preprocess_train_data(word2vec_model, train_lyrics, midi_data, get_midi_simple_embedding)
simple_model = create_lstm_model(word2vec_model, melody_embedding_features=128, LSTM_layer_size=200)
simple_model.summary()

simple_model.compile(**compilation_params)
simple_model_checkpoint_filepath = './checkpoint/simple_model'
os.makedirs(simple_model_checkpoint_filepath, exist_ok=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=simple_model_checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
log_dir = "./logs/simple_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [tensorboard_callback, model_checkpoint_callback]

#  Training the simple_model
simple_model_loss = simple_model.fit(input_generator(X, Y), epochs=30, steps_per_epoch=len(train_lyrics), callbacks=callbacks)




################################################
############## Complex Model ###################
################################################
#  Preprocessing the data for the complex_model
X, Y = preprocess_train_data(word2vec_model, train_lyrics, midi_data, get_midi_complex_embedding)
complex_model = create_lstm_model(word2vec_model, melody_embedding_features=256, LSTM_layer_size=200)
complex_model.summary()

complex_model.compile(**compilation_params)
complex_model_checkpoint_filepath = './checkpoint/complex_model'
os.makedirs(complex_model_checkpoint_filepath, exist_ok=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=complex_model_checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

log_dir = "./logs/complex" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [tensorboard_callback, model_checkpoint_callback]

#  Training the complex_model
complex_model_loss = complex_model.fit(input_generator(X, Y), epochs=30, steps_per_epoch=len(train_lyrics), callbacks=callbacks)



################################################
############## Generating Songs ################
################################################

#simple_model = tf.keras.models.load_model(simple_model_checkpoint_filepath)
#complex_model = tf.keras.models.load_model(complex_model_checkpoint_filepath)

test_lyrics = load_lyrics('./lyrics_test_set.csv')
test_midi_data = get_midi_data(test_lyrics)

#  Preprocessing the test-data for both models
test_simple_X = preprocess_test_data(word2vec_model, test_lyrics, test_midi_data, get_midi_simple_embedding)
test_complex_X = preprocess_test_data(word2vec_model, test_lyrics, test_midi_data, get_midi_complex_embedding)
simple_generated_songs = []
complex_generated_songs = []

#  Generating songs using the simple_model
for song in test_simple_X:
    simple_generated_song = generate_song(RNN_model=simple_model, embedding_model=word2vec_model, melody_embeddings=song[1], initial_word=song[0])
    simple_generated_songs.append({'original_song': song[2], 'first_word': song[0], 'generated_song': simple_generated_song})

#  Generating songs using the complex_model
for song in test_complex_X:
    complex_generated_song = generate_song(RNN_model=complex_model, embedding_model=word2vec_model, melody_embeddings=song[1], initial_word=song[0])
    complex_generated_songs.append({'original_song': song[2], 'first_word': song[0], 'generated_song': complex_generated_song})



########################################################
#################### EVALUATION ########################
########################################################
simple_word2vec_mean, simple_word2vec_std, simple_doc2vec_mean, simple_doc2vec_std =\
    evaluate_generated_songs(word2vec_model, test_lyrics, simple_generated_songs)
complex_word2vec_mean, complex_word2vec_std, complex_doc2vec_mean, complex_doc2vec_std = \
    evaluate_generated_songs(word2vec_model, test_lyrics, complex_generated_songs)

generate_evaluation_figure(simple_word2vec_mean, simple_word2vec_std,
                           complex_word2vec_mean, complex_word2vec_std, "Word2Vec")
generate_evaluation_figure(simple_doc2vec_mean, simple_doc2vec_std,
                           complex_doc2vec_mean, complex_doc2vec_std, "Doc2Vec")

