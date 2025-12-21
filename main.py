from smart_open import open

text = open('https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt', 'r').read()

print(len(text))

print(text[:1000])