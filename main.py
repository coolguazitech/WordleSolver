from sys import argv
from collections import Counter
import os
import numpy as np
from numpy.random import normal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

VERSION = '1.3'

N_CHANCES = 6
WORD_LENGTH = 5
SCORING_RANGE = 5
VOWELS = {'a', 'e', 'i', 'o', 'u'}
SCORERS = ['Pipeline([("ss", StandardScaler()),("mlp", MLPClassifier(hidden_layer_sizes=(8,16,32), max_iter=2000, tol=1e-6, learning_rate="adaptive"))])']
CHAR_2_RATE = {'A': 10, 'B': 5, 'X': 1}
STOP_WORDS_PATH = 'stop_words.txt'
CORPUS_PATH = 'corpus.txt'
GUESS_RESULTS_PATH = os.path.join('ml', f'guess_results_{WORD_LENGTH}l_{SCORING_RANGE}r.data')

def extend_corpus(src_path, dst_path=CORPUS_PATH):
    """Append words to corpus from source file."""
    src_f = open(src_path, 'r')
    dst_f = open(dst_path, 'a')
    while True:
        w = src_f.readline()
        if not w: break
        dst_f.write(w)
    src_f.close()
    dst_f.close()

def add_stop_word(word, dst_path=STOP_WORDS_PATH):
    """Add a stop word."""
    with open(dst_path, 'a') as f:
        f.write(word + '\n')

def vectorize_word(word):
    """Vectorize an iterable of word."""
    return list(map(ord, word))

def insert_results(results_with_scores, dst_path=GUESS_RESULTS_PATH):
    """Insert game results into archive for online learning.

    Parameters
    ----------
    results_with_scores : 2D array, where nRow, nCol = (USED CHANCES, WORD_LENGTH + 1)
      Each row would be [n1, n2, ..., nk, score] where ni denotes the ASCII code of gi,
      k denotes WORD_LENGTH, and the integer score demonstrates how good this guess is. 
      The score is in [0, RATING_RANGE].

    """
    m, n = len(results), len(results[0]) // 2
    dst_f = open(dst_path, 'a')
    for r in results_with_scores:
        dst_f.write(','.join(map(str, r)) + '\n')
    dst_f.close()

def score(results):
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
      k denotes WORD_LENGTH, and the integer score demonstrates how good this guess is. 
      The score is in [0, RATING_RANGE].

    """
    _, n = len(results), len(results[0]) // 2
    results_with_scores = [vectorize_word(r[:n]) + [0] for r in results]

    # Current criterion (gini + rating):
    max_rate = max(CHAR_2_RATE.values()) * WORD_LENGTH
    for i, r in enumerate(results):
        cnt = Counter(r[n:])
        impurity = 1 - sum((cnt[c] / WORD_LENGTH) ** 2 for c in cnt)
        spots_rate = sum(CHAR_2_RATE[c] * cnt[c] for c in cnt) / max_rate
        _score = int((0.4 * (1 - impurity) ** 0.5 + 0.6 * spots_rate) * SCORING_RANGE)
        results_with_scores[i][-1] = _score

    return results_with_scores

class Wordlist:
    def __init__(self, enable_AI=False):
        self._size = 0
        self._words = []
        self._scorers = []
        if enable_AI:
            self._get_scorers()
            self._train_scorers()
        self._load_words()
        self._rearrange_words()

    @property
    def size(self):
        return self._size
    
    @property
    def words(self):
        return self._words
    
    def _rearrange_words(self):
        """Sort by the sorting criteria."""
        self._words.sort(key=self._sort_criteria)
        
    def _load_words(self, src_path=CORPUS_PATH, stop_words_path=STOP_WORDS_PATH):
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
        # 0 to RATING_RANGE points for each criterion
        priority = 0

        # Criterion 1: the number of vowels in the word
        priority += int(sum(c in VOWELS for c in word) * SCORING_RANGE / WORD_LENGTH) 

        # Criterion 2: the diversity of letters in the word
        priority += int((len(set(word)) - 1) * SCORING_RANGE / (WORD_LENGTH - 1)) 

        # Criterion 3: the AI scorers
        _word = vectorize_word(word)
        for scorer in self._scorers:
            priority += scorer.predict([_word])[0] 

        return priority * normal(loc=1.0, scale=0.2)

    def _get_scorers(self):
        """Get instantiated non-trained scorers."""
        self._scorers = list(map(eval, SCORERS))

    def _train_scorers(self, src_path=GUESS_RESULTS_PATH):
        """Load dataset and use it to train scorers."""
        print('Get intelligence...')
        # load data
        src_f = open(src_path, 'r')
        data = [r[:-1].split(',') for r in src_f.readlines()]
        data = np.array(data, dtype=np.int)
        X, y = data[:, :-1], data[:, -1]
        src_f.close()

        # train estimators
        for scorer in self._scorers:
            scorer.fit(X, y)

class Solver:
    def __init__(self, enable_AI=False):
        self.wordlist = Wordlist(enable_AI)
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
    # print version
    print(f'Version {VERSION}')

    # initialization
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

    # loop
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

    # score the results and insert it into archive
    results_with_scores = score(results)
    insert_results(results_with_scores)
    print('Thank you for using! We expect your feedback.')
            
