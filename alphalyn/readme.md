# AlphaZero
**AlphaZero** is a reinforcement learning algorithm that masters games like chess, shogi, and Go entirely through **self-play**, without any human knowledge or data. It learns to play by starting from random moves and improving over time by playing millions of games against itself.

The algorithm has three main components:

- **A deep neural network** that is trained to predict the best move to make from a given game state (the policy) and the likely winner of the game (the value).
- **Monte Carlo Tree Search (MCTS)**, an advanced search algorithm that uses the neural network's predictions to explore possible moves and build a game tree, helping the AI decide on the most promising action.
- **Self-play**, the engine of AlphaZero's learning. The algorithm repeatedly plays games against itself. Each game provides new data (game states, chosen moves, and the final winner) to train the neural network, continuously improving its policy and value predictions.
