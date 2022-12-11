from sys import argv
from collections import Counter
from random import shuffle

CHANCES = 6
WORD_LENGTH = 5
VOWELS = {'a', 'e', 'i', 'o', 'u'}

def extend_corpus(src_path, dst_path='corpus.txt'):
    """Append words to corpus from source file."""
    src_f = open(src_path, 'r')
    dst_f = open(dst_path, 'a')
    while True:
        w = src_f.readline()
        if not w: break
        dst_f.write(w)
    src_f.close()
    dst_f.close()

class Wordlist:
    def __init__(self, word_length=5):
        self._size = 0
        self._words = []
        self.word_length = word_length
        self.criterion = lambda w: sum(c in VOWELS for c in w)

    @property
    def size(self):
        return self._size
    
    @property
    def words(self):
        return self._words
    
    def rearrange_words(self):
        """Rearrange words by criterion."""
        shuffle(self._words)
        self._words.sort(key=self.criterion)
        
    def load_words(self, src_path='corpus.txt'):
        """Load words with its size from corpus."""
        words = set()
        with open(src_path, 'r') as f:
            while True:
                w = f.readline()
                if not w: break
                w = w.strip().lower()
                if w.isalpha() and len(w) == self.word_length:
                    words.add(w)
        self._size, self._words = len(words), list(words)

class Solver:
    def __init__(self, wordlist, word_length=5):
        self.wordlist = wordlist
        self.size = wordlist.size
        self.words = wordlist.words
        self.word_length = word_length
        
    def reset(self):
        """Reset the solver."""
        self.__init__(self.wordlist)
        
    def next_word(self):
        """Pop out the next possible word."""
        if not self.size:
            raise Exception('Solver crashed because there didn\'t remain any words. Please extend the corpus.')
        self.size -= 1
        return self.words.pop()
    
    def _judge(self, guess, ans):
        """Determine how the guess and the answer match."""
        cnt = Counter(ans)
        judgment = ['X'] * self.word_length
        for i in range(self.word_length):
            if guess[i] == ans[i]: 
                judgment[i] = 'A'
                cnt[guess[i]] -= 1

        for i in range(self.word_length):
            if cnt[guess[i]]: 
                judgment[i] = 'B'
                cnt[guess[i]] -= 1

        return ''.join(judgment)
    
    def update_words(self, guess, judgment):
        """Delete the impossible words."""
        remained_words = []
        for w in self.words:
            if self._judge(guess, w) == judgment:
                remained_words.append(w)
            else: self.size -= 1
        self.words = remained_words

if __name__ == '__main__':
    if len(argv) != 1 and len(argv) != 3 or len(argv) == 3 and not (argv[1] == '-u' or argv[1] == '--update'):
        raise Exception('Provide no arguments and start solver. Or, use "[-u][--update] filepath" to extend the corpus.')
    elif len(argv) != 1:
        print('Update the corpus...')
        extend_corpus(argv[2])
    print('Load words from corpus...')
    wl = Wordlist(WORD_LENGTH)
    wl.load_words()
    wl.rearrange_words()
    solver = Solver(wl, WORD_LENGTH)
    print('Start!')

    for _ in range(CHANCES):
        while True:
            guess = solver.next_word()
            while True:
                inputs = input(f'I guess the word is "{guess}". Please give me the hint ("ABBAX" or "!"): ').upper()
                if inputs == '!' or len(inputs) == WORD_LENGTH and inputs.isalpha() and set(inputs) <= set("ABX") : break
                print('Invalid inputs, please try again.')
            if inputs != '!': break

        if inputs == 'A' * WORD_LENGTH:
            print('You win!')
            break

        solver.update_words(guess, inputs)
    else: print('Sorry I can\'t solve it...')
            