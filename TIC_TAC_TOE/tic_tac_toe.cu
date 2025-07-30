#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

__device__ void make_move(char* board, int idx, char symbol) {
    if (board[idx] == ' ') board[idx] = symbol;
}

__device__ int find_winning_move(char* board, char symbol) {
    int win_positions[8][3] = {
        {0,1,2},{3,4,5},{6,7,8}, // rows
        {0,3,6},{1,4,7},{2,5,8}, // cols
        {0,4,8},{2,4,6}          // diags
    };
    for (int i = 0; i < 8; i++) {
        int a = win_positions[i][0], b = win_positions[i][1], c = win_positions[i][2];
        if (board[a] == symbol && board[b] == symbol && board[c] == ' ') return c;
        if (board[a] == symbol && board[c] == symbol && board[b] == ' ') return b;
        if (board[b] == symbol && board[c] == symbol && board[a] == ' ') return a;
    }
    return -1;
}

__global__ void playerA_turn(char* board) {
    int idx = find_winning_move(board, 'X');
    if (idx == -1) idx = find_winning_move(board, 'O'); // block O
    if (idx == -1) {
        for (int i = 0; i < 9; i++) {
            if (board[i] == ' ') {
                idx = i;
                break;
            }
        }
    }
    make_move(board, idx, 'X');
}

__global__ void playerB_turn(char* board, int seed) {
    int idx = -1;
    for (int i = 0; i < 50 && idx == -1; i++) {
        int r = (seed + i) % 9;
        if (board[r] == ' ') {
            idx = r;
        }
    }
    if (idx == -1) {
        for (int i = 0; i < 9; i++) {
            if (board[i] == ' ') {
                idx = i;
                break;
            }
        }
    }
    make_move(board, idx, 'O');
}

__host__ void print_board(const char* board) {
    for (int i = 0; i < 9; i++) {
        printf("%c ", board[i]);
        if (i % 3 == 2) printf("\n");
    }
    printf("\n");
}

__host__ void save_board(FILE* f, const char* board, int turn) {
    fprintf(f, "Turn %d:\n", turn);
    for (int i = 0; i < 9; i++) {
        fprintf(f, "%c ", board[i]);
        if (i % 3 == 2) fprintf(f, "\n");
    }
    fprintf(f, "\n");
}

__host__ bool check_win(const char* board, char symbol) {
    int win_positions[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };
    for (int i = 0; i < 8; i++) {
        if (board[win_positions[i][0]] == symbol &&
            board[win_positions[i][1]] == symbol &&
            board[win_positions[i][2]] == symbol)
            return true;
    }
    return false;
}

__host__ bool board_full(const char* board) {
    for (int i = 0; i < 9; i++) if (board[i] == ' ') return false;
    return true;
}

int main() {
    char host_board[9];
    for (int i = 0; i < 9; i++) host_board[i] = ' ';
    char* dev_board;
    cudaMalloc(&dev_board, 9 * sizeof(char));
    cudaMemcpy(dev_board, host_board, 9 * sizeof(char), cudaMemcpyHostToDevice);

    FILE* f = fopen("replay.txt", "w");

    int turn = 0;
    srand(time(NULL));
    while (true) {
        if (turn % 2 == 0) {
            playerA_turn<<<1, 1>>>(dev_board);
        } else {
            int seed = rand();
            playerB_turn<<<1, 1>>>(dev_board, seed);
        }

        cudaMemcpy(host_board, dev_board, 9 * sizeof(char), cudaMemcpyDeviceToHost);
        print_board(host_board);
        save_board(f, host_board, turn);

        if (check_win(host_board, 'X')) {
            fprintf(f, "Player A (X) wins!\n");
            printf("Player A (X) wins!\n");
            break;
        } else if (check_win(host_board, 'O')) {
            fprintf(f, "Player B (O) wins!\n");
            printf("Player B (O) wins!\n");
            break;
        } else if (board_full(host_board)) {
            fprintf(f, "It's a draw!\n");
            printf("It's a draw!\n");
            break;
        }

        turn++;
    }

    fclose(f);
    cudaFree(dev_board);
    return 0;
}
