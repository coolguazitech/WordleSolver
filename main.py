from sys import argv
from collections import Counter
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from numpy.random import normal

VERSION = '1.0.0'

N_CHANCES = 6
WORD_LENGTH = 5
RATING_RANGE = 3
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

def add_stop_word(word, dst_path='stop_words.txt'):
    """Add a stop word."""
    with open(dst_path, 'a') as f:
        f.write(word + '\n')

def insert_results(results, dst_path=os.path.join('ml', 'guess_results_5.data')):
    """Insert game results into archive for online learning.

    Parameters
    ----------
    results : 2D array, where nRow, nCol = (USED CHANCES, WORD_LENGTH * 2)
      Each row would be [g1, g2, ..., gk, r1, r2, ..., rk] where gi denotes the guessed letter;
      ri denotes the result letter, and k denotes WORD_LENGTH.

    """
    m, n = len(results), len(results[0]) // 2
    results_with_scores = _score(results)
    dst_f = open(dst_path, 'a')
    for r in results_with_scores:
        dst_f.write(','.join(map(str, r)) + '\n')
    dst_f.close()

def _score(results):
    """Define scoring criteria for guess results.

    Parameters
    ----------
    results : 2D array, where nRow, nCol = (USED CHANCES, WORD_LENGTH * 2)
      Each row would be [g1, g2, ..., gk, r1, r2, ..., rk] where gi denotes the guessed letter;
      ri denotes the result letter, and k denotes WORD_LENGTH.

    Returns
    -------
    results_with_scores : 2D array, where nRow, nCol = (USED CHANCES, WORD_LENGTH + 1)
      Each row would be [n1, n2, ..., nk, score] where ni denotes the ASCII code of gi,
      k denotes WORD_LENGTH, and the score demonstrates how good this guess is.

    """
    m, n = len(results), len(results[0]) // 2
    results_with_scores = [list(map(ord, r[:n])) + [0] for r in results]

    # Current criterion:
    char_2_point = {'A': 2, 'B': 1, 'X': 0}
    for i, r in enumerate(results):
        cnt = Counter(r[n:])
        results_with_scores[i][-1] = (N_CHANCES - i) * sum(char_2_point[c] * cnt[c] for c in cnt)

    return results_with_scores

class Wordlist:
    def __init__(self, enable_AI=False):
        self._size = 0
        self._words = []
        self.enable_AI = enable_AI

    @property
    def size(self):
        return self._size
    
    @property
    def words(self):
        return self._words
    
    def rearrange_words(self):
        """Sort by the sorting criteria."""
        if self.enable_AI:
            self.estimator = self._get_learned_regressor()
        self._words.sort(key=self._sort_criteria)
        
    def load_words(self, src_path='corpus.txt', stop_words_path='stop_words.txt'):
        """Load and count the valid words from corpus except for stop words."""
        print('Load words from corpus...')
        words, stop_words = set(), set()
        words_f = open(src_path, 'r')
        stop_words_f = open(stop_words_path, 'r')

        while True:
            w = stop_words_f.readline()
            if not w: break
            w = w.strip().lower()
            if w.isalpha() and len(w) == WORD_LENGTH:
                stop_words.add(w)

        while True:
            w = words_f.readline()
            if not w: break
            w = w.strip().lower()
            if w.isalpha() and len(w) == WORD_LENGTH and w not in stop_words:
                words.add(w)

        words_f.close()
        stop_words_f.close()
        self._size, self._words = len(words), list(words)

    def _sort_criteria(self, word):
        """Determine how words are sorted.

        Retruns
        -------
        priority : float
          a number randomized with Gaussian noise 

        """
        # 1 to RATING_RANGE points for each criterion
        priority = 0

        # Criterion 1: the number of vowels in the word
        priority += int(sum(c in VOWELS for c in word) * (RATING_RANGE - 1) / WORD_LENGTH) + 1

        # Criterion 2: the diversity of letters in the word
        priority += int((len(set(word)) - 1) * (RATING_RANGE - 1) / (WORD_LENGTH - 1)) + 1

        # Criterion 3: the AI scorer
        if self.enable_AI:
            maximum = 2 * N_CHANCES * WORD_LENGTH
            _word = list(map(ord, word))
            priority += int(self.estimator.predict([_word])[0] * (RATING_RANGE - 1) / maximum) + 1

        return priority * normal(loc=1.0, scale=0.1)

    def _get_learned_regressor(self, src_path=os.path.join('ml', 'guess_results_5.data')):
        """Load the data and use it to train a regression model, then return the model."""
        # load data
        src_f = open(src_path, 'r')
        pipe_lr = Pipeline([('ss', StandardScaler()), ('lr', LinearRegression())])
        data = [r[:-1].split(',') for r in src_f.readlines()]
        src_f.close()

        # train model
        data = np.array(data, dtype=np.float)
        X, y = data[:, :-1], data[:, -1]
        pipe_lr.fit(X, y)

        return pipe_lr

class Solver:
    def __init__(self, enable_AI=False):
        self.wordlist = Wordlist(enable_AI)
        self.wordlist.load_words()
        self.wordlist.rearrange_words()
        self.size = self.wordlist.size
        self.words = self.wordlist.words
        self.enable_AI = enable_AI
        
    def reset(self):
        """Reset the solver."""
        self.__init__(self.enable_AI)
        
    def next_word(self):
        """Pop out the next possible word."""
        if not self.size:
            raise Exception('Solver crashed because there didn\'t remain any words. Please extend the corpus.')
        self.size -= 1
        return self.words.pop()
    
    def _judge(self, guess, ans):
        """Determine how the guess and the answer match."""
        cnt = Counter(ans)
        judgment = ['X'] * WORD_LENGTH
        for i in range(WORD_LENGTH):
            if guess[i] == ans[i]: 
                judgment[i] = 'A'
                cnt[guess[i]] -= 1

        for i in range(WORD_LENGTH):
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
    # initialize corpus
    if len(argv) != 1 and len(argv) != 3 or len(argv) == 3 and not (argv[1] == '-u' or argv[1] == '--update'):
        raise Exception('Provide no arguments and start solver. Or, use "[-u <path>][--update <path>]" to extend the corpus.')
    elif len(argv) != 1:
        print('Update the corpus...')
        extend_corpus(argv[2])
    
    # check if the AI features are enabled
    enable_AI = False
    ans = input('Do you want to enable AI features? [y/n] ')
    if ans.lower() == 'y':
        enable_AI = True

    # initialize the solver
    solver = Solver(enable_AI=enable_AI)
    print('Start!')

    results = []
    for _ in range(N_CHANCES):
        while True:
            guess = solver.next_word()
            while True:
                inputs = input(f'I guess the word is "{guess}". Please give me the hint ("ABBAX" or "!"): ').upper()
                if inputs == '!' or len(inputs) == WORD_LENGTH and inputs.isalpha() and set(inputs) <= set("ABX"): 
                    break
                print('Invalid inputs, please try again.')
            if inputs == '!': 
                add_stop_word(guess)
                continue
            else: break
                
        results.append([*guess, *inputs])
        if inputs == 'A' * WORD_LENGTH:
            print('You win!')
            break

        solver.update_words(guess, inputs)
    else: print('Sorry I can\'t solve it...')
    insert_results(results)
    print('Thank you for using! We expect your feedback.')
            
