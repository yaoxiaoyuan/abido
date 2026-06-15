# AlphaZero
**AlphaZero** is a reinforcement learning algorithm that masters games like chess, shogi, and Go entirely through **self-play**, without any human knowledge or data. It learns to play by starting from random moves and improving over time by playing millions of games against itself.

The algorithm has three main components:

- **A deep neural network** that is trained to predict the best move to make from a given game state (the policy) and the likely winner of the game (the value).
- **Monte Carlo Tree Search (MCTS)**, an advanced search algorithm that uses the neural network's predictions to explore possible moves and build a game tree, helping the AI decide on the most promising action.
- **Self-play**, the engine of AlphaZero's learning. The algorithm repeatedly plays games against itself. Each game provides new data (game states, chosen moves, and the final winner) to train the neural network, continuously improving its policy and value predictions.

While the original AlphaZero framework by DeepMind revolutionized AI, it is notoriously resource-heavy, requiring massive computational power for self-play and training.
In open-source implementations (like Leela Chess Zero (Lc0), KataGo for Go, and Crazy Ara) and subsequent academic research, developers have introduced critical optimizations to make AlphaZero-like networks faster, more sample-efficient, and runnable on consumer hardware.
These optimizations span the algorithm, neural network architecture, and engineering infrastructure.
- 1. Algorithmic Optimizations (MCTS & Search)
The core bottleneck of AlphaZero is Monte Carlo Tree Search (MCTS). Several algorithmic changes have massively accelerated search efficiency:
Continuous Updating (Asynchronous Training): Original AlphaGo Zero used an evaluation phase where a "challenger" network had to beat the "anchor" network (e.g., win 55% of games) to replace it. AlphaZero optimized this by removing the evaluation phase completely. The network is updated continuously, and the latest weights are immediately used for self-play, accelerating the feedback loop.
Virtual Loss: To utilize multi-threading during MCTS, multiple threads must traverse the tree simultaneously. To prevent different threads from exploring the exact same path before the neural network returns an evaluation, a virtual loss is temporarily added to a node's visit count when a thread visits it. This forces other threads to explore alternative moves in parallel.
Transposition Tables and Graph-Structured Search: Classic MCTS builds a tree. However, board games often arrive at the same position via different move orders (transpositions). Modern optimizations turn the search tree into a Directed Acyclic Graph (DAG), allowing different branches to share identical sub-nodes, saving massive amounts of memory and NN evaluations.
Tree Reuse: Instead of throwing away the search tree after every move, the subtree corresponding to the chosen move is retained. The visit counts and evaluations are carried over to the next move, drastically reducing the search depth required for subsequent turns.
First Play Urgency (FPU): When a node has unvisited children, standard MCTS doesn't know how to value them. FPU optimizes this by assigning a clever heuristic value to unvisited moves (usually based on the parent's value minus a margin) so the engine doesn't waste time exploring obviously bad moves just because their visit count is zero.
- 2. Structural & Architectural Optimizations (The Neural Network)
The deep residual network (ResNet) is the heaviest component of the system. Optimizations here aim to squeeze more performance out of fewer parameters.
Dual-Headed Network: A foundational AlphaZero optimization over early AlphaGo models was merging the Policy Network (which moves to pick) and the Value Network (who is winning) into a single body with two output "heads". This cut the required forward-pass computations nearly in half.
Squeeze-and-Excitation (SE) Blocks: Modern implementations (like Lc0 and KataGo) insert SE layers into the residual blocks. SE blocks globally pool spatial information, allowing the network to recognize relationships across the entire board much earlier without needing an excessively deep network.
MobileNet / EfficientNet Backbones: Replacing heavy convolutional layers with depthwise separable convolutions allows engines to run on lower-end GPUs or mobile devices with minimal loss in Elo rating.
ViT vs. CNN Hybridization: While pure Vision Transformers (ViTs) are often too slow for the step-by-step nature of MCTS, hybrid architectures (like combining MobileNet with NextViT) have been used to beat standard AlphaZero baselines by capturing long-range board state dependencies more efficiently.
- 3. Engineering & Hardware-Level Optimizations
Getting the software to talk to hardware with zero latency is where massive speedups are made.
NN Evaluation Batching: A single GPU forward pass for one board state is highly inefficient. Optimizations involve pausing MCTS threads when they hit a leaf node, gathering board states from dozens of parallel games or search paths, and sending them to the GPU as a single gathered batch.
Quantization (FP16, INT8, and TensorRT): AlphaZero was trained using FP32 (32-bit floating-point precision). For inference and search, networks are optimized using FP16 or INT8 precision. Using NVIDIA TensorRT or Apple CoreML to quantize weights allows the engine to utilize Tensor Cores, often speeding up position evaluation by 5x to 10x with negligible strength loss.
Playout Cap Randomization: During training, AlphaZero does a full search (e.g., 800 playouts) for every single move. An optimization introduced in newer setups dictates doing a full search on only ~10% of moves, and a tiny search (e.g., 10–20 playouts) on the rest. This speeds up training data generation by up to 700% while retaining high-quality data.
- 4. Domain-Specific Optimizations (Data & Rules)
Global Feature Pooling: In games like Go (KataGo), explicit features like "how many liberties does this group have?" or "who owns this territory?" are fed into the network. Giving the network elementary game concepts dramatically lowers the number of games it needs to play to reach superhuman status.
Symmetry Exploitation (Dihedral Group): For games like Go or Reversi, the board can be rotated or mirrored without changing the game state rules. By randomly rotating the board inputs during self-play data generation, the model learns invariants much faster, effectively multiplying the training data for free. (Note: This is tougher in Chess due to asymmetrical moves like castling or pawn direction, though some implementations still mirror the board horizontally).

