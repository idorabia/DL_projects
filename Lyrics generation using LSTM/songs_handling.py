import pickle
with open("simple generated songs", 'rb') as s:
    simple_generated_songs = pickle.load(s)
with open("complex generated songs", 'rb') as c:
    complex_generated_songs = pickle.load(c)

complex_songs = []
for complex_song in complex_generated_songs:
    complex_song = " ".join(complex_song["generated_song"])
    complex_song = complex_song.replace("eos", "")
    complex_song = complex_song.replace("eol", "&")
    complex_songs.append(complex_song)
simple_songs = []
for simple_song in simple_generated_songs:
    simple_song = " ".join(simple_song["generated_song"])
    simple_song = simple_song.replace("eos", "")
    simple_song = simple_song.replace("eol", "&")
    simple_songs.append(simple_song)
print("dddd")