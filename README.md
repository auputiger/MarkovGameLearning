# rl_project3
name: Jesse Stills
<br>
gatech username: jstills3
<br>
<br>
Repository for project 3 of Reinforcement Learning.

## Project structure
* rl_project3
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