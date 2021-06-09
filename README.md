# RLChopsticks
Applying reinforcement learning techniques to the hand game of [chopsticks](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)).

# How the Model Plays
For simplicity, the model consideres each *state* within the game independent (that is, it doesn't learn from the opponent's technique).

With this in mind, we can encode each state with four numbers:

1. The # of fingers on the model's left hand
2. The # of fingers on the model's right hand
3. The # of fingers on the opponent's left hand
4. The # of fingers on the opponents's right hand

Note that the order of each player's hands is irrelevant. Taking advantage of this symmetry, there are 225 possible states.

Note that some states are more optimal than others (e.g. having four fingers on each hand is worse than one finger on each).

Consider each state as within a 5x5x5x5 *state space* (that is, a four-dimensional tensor), which we can use to map each state, S, to the probability, P<sub>S</sub>, of the model winning from S (e.g. if the opponent is missing both hands, said probability would be 1; if the model is missing both hands; said probability would be 0). Because the model can move from any state to an enumerable number of others, it can play optimally by always choosing the new state with the highest said probability.

# How the Model Trains
We can empirically approximate P<sub>S</sub> for every state S by initializing two models with random probabilistic mappings (that is, horrible understandings of how to play). Then, they'll play against each other repeatedly, each time updating P<sub>S</sub> for each state S in which they were. Taking a lesson from the human experience, they can't learn without making mistakes. Therefore, when training, they'll sort each possible move S by P<sub>S</sub> and choose the first for which a Bernoulli distribution sample equals one.

# Usage

`$ cargo build`

`$ cargo run`

Input each move as "A B C D" where A and B are the # of fingers on your hand and B and C are the # of fingers on your opponent's hand.
