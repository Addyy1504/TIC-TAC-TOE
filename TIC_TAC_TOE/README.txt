# 🕹️ GPU vs GPU Tic-Tac-Toe (CUDA Project)

This project simulates a **Tic-Tac-Toe game** where **two GPU kernels compete** as players using distinct strategies. The game alternates between kernel launches for each player's move, with the board state stored and updated in global device memory.

## ✅ Project Description

- **Player A** uses a **greedy strategy**:
  - Tries to win in one move.
  - If not possible, tries to block the opponent.
  - Otherwise, picks the first available cell.

- **Player B** uses a **random strategy**:
  - Picks a random empty cell from the board.

Each move is saved in a `replay.txt` file, which can later be visualized as a game replay in PowerPoint or video format.

---

## 📁 Files

| File | Description |
|------|-------------|
| `tic_tac_toe.cu` | Main CUDA code: game logic + GPU kernels |
| `Makefile`       | Builds the project using `nvcc` |
| `replay.txt`     | Game log: board state after each move |

---

## 👨‍💻 Code Structure

- The host alternates launching:
  - `playerA_turn <<<>>>` — Player A's GPU turn
  - `playerB_turn <<<>>>` — Player B's GPU turn

- The board is a **1D array of 9 `char` values** (`'X'`, `'O'`, `' '`).

- The host:
  - Reads board state from device.
  - Prints to terminal and appends to `replay.txt`.

---

## 🔁 Kernels

- `__global__ void playerA_turn(char* board);`  
  Greedy strategy: win/block/pick.

- `__global__ void playerB_turn(char* board, int seed);`  
  Random strategy: pick empty at random.

---

## 🧪 How to Compile and Run

### 🔧 Compile
```bash
make
