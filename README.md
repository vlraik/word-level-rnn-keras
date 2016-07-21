# word-level-rnn
Word Level RNN for text generation using keras.

This is a project with a goal to make a Word Level Text generation using keras on Theano Framework.
The code is based on the sample text generation character level code provided as a sample example at
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

#Read This:
Since the code is done at word level instead of character level, it needs more epochs than the character 
level text generation counter part, as well as much more memory because it has to be trained on thousands
of words, (unlike character level code where it's only trained on 50-100 different characters (assuming 
ASCII). 

Word level RNN text generation should in theory be more coherent then the Character level text generation
at lesser amount of training.
Suggestions and criticism are welcome to optimising and improve the code. thanks!
