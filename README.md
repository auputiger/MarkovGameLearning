# Markov Game Learning Convergence Experiment

## Description
The purpose of this project was to test the convergence of four different types of 
learning algorithms in a simple zero sum markov game (4x2 grid soccer game with 2 players).
Each algorithm simulates 1,000,000 games and checks if the Q-value of a particular state
converges. The algorithms tested are: Q-Learning, Friend Q-Learning, Foe Q-Learning, and Correlated Q-Learning.
Foe-Q and Correlated-Q use linear programming.


## Project structure
* MarkovGameLearning
    * soccer
        * actions.py
        * player.py
        * soccer_game.py
        * solver.py
        * state.py
    * main.py
    * README.md

## How To Run
1. Install Python 3.5

2. Please install cvxopt and the required dependencies
   http://cvxopt.org/install/

3. Using Python 3.5, run `main.py`. Some logging is printed to standard out to give me idea how far along each test is.

4. Results for each type of learning are printed to files `q-learning.csv`, `friend-q.csv`, `foe-q.csv`, `ce-q.csv`. 
   Results are a csv with the following format: time-step,q-value-diff,pre-q-value,post-q-value,action/joint-action probabilities (foe-q & ce-q)