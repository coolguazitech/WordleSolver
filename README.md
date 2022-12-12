# Wordle Solver AI
### Version 1.0.0

### Brief
I create a solver featuring AI for the pop game WORDLE. With this program, you can:
1. Just give it a hint each round when it guesses a word, and you'll be able to solve puzzles quickly.

1. Extend the corpus from your word list file if needed. 

### Prerequisite
To run the code correctly, you need to install *scikit-learn* and *numpy* packages using

`pip install -U scikit-learn`

`pip install numpy`

### Usage
Enter the folder where `main.py` is located, and

* run the solver directly with

    ```python main.py```

* extend the corpus and run the solver with

    ```python main.py [-u <path>][--update <path>]```

Tip: Infrequently used words may lead to bad performance.
    

### Interactions
First and foremost, you can decide whether to use AI features or not. If yes, the solver will use it to guess the word. In each round, the solver will first guess a word. At this time, you have to give it a **hint** according to the result (the default is green, yellow and gray spots), and proceed in this way until the end of the game. E.g: If the result shows like this, 

![](https://github.com/coolguazitech/WordleSolver/blob/main/assets/wordle_eg.jpg)

then the hint you'll give it is 'XBXXA', where 'X' stands for gray, 'A' for green, and 'B' for yellow.
If the result shows there is no such word, enter one '!'. In the case the chances will not decrement.
If the program fails in any way, consider exing the corpus.
