/*
 * chess.h - Complete chess engine for PufferLib self-play RL
 *
 * Single-header chess environment supporting two agents (White and Black)
 * per game instance. Designed for high-performance reinforcement learning
 * with PufferLib Ocean conventions.
 *
 * Two-phase action system (97 actions):
 *   Phase 0: Pick a piece (action 0-63 = board square)
 *   Phase 1: Pick destination (0-63) or promotion (64-95)
 *   Action 96: PASS (valid when it's NOT this player's turn)
 *
 * Board representation: array of 64 int8_t values
 *   square = row * 8 + col
 *   row 0 = rank 1 (White back rank), row 7 = rank 8 (Black back rank)
 *   a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
 *
 * Pieces: 0=empty, 1-6=White(P,N,B,R,Q,K), 7-12=Black(P,N,B,R,Q,K)
 */

#ifndef CHESS_H
#define CHESS_H

#include <stdint.h>
#include <string.h>

// ============================================================================
// Constants
// ============================================================================

#define CHESS_NUM_ACTIONS 97       // 64 squares + 32 promotions + 1 pass
#define CHESS_PASS_ACTION 96
#define CHESS_MAX_MOVES 256

// Observation layout (per player):
//   64 board + 2 side + 4 castling + 1 ep + 2 phase + 64 selected_piece
//   + 64 valid_pieces + 64 valid_dests + 32 valid_promos
//   + 1 self_check + 1 opp_check + 1 rule50 + 1 pass_valid
// Total = 301
#define CHESS_OBS_SIZE 301

#define EMPTY 0
#define WP 1
#define WN 2
#define WB 3
#define WR 4
#define WQ 5
#define WK 6
#define BP 7
#define BN 8
#define BB 9
#define BR 10
#define BQ 11
#define BK 12

// Castling right bits
#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

// Game end codes
#define GAME_ONGOING 0
#define GAME_CHECKMATE 1
#define GAME_STALEMATE 2
#define GAME_FIFTY_MOVE 3
#define GAME_INSUFFICIENT 4

// ============================================================================
// Structures
// ============================================================================

typedef struct Log {
    float episode_length;
    float episode_return;
    float white_wins;
    float black_wins;
    float draws;
    float illegal_moves;
    float n;
} Log;

// Per-player two-phase state
typedef struct {
    int pick_phase;           // 0=pick piece, 1=pick destination
    int selected_square;      // square picked in phase 0 (-1 if none)
    int valid_dest_moves[CHESS_MAX_MOVES]; // legal moves from selected_square (encoded as from*64+to)
    int valid_dest_count;
} PhaseState;

typedef struct ChessEnv {
    unsigned char* observations;
    void* actions;
    int action_itemsize;  /* 4 (int32) or 8 (int64) */
    float* rewards;
    unsigned char* terminals;
    Log log;

    int obs_stride;

    int8_t board[64];
    int current_player;         // 0=White, 1=Black
    uint8_t castling_rights;    // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
    int en_passant_square;      // -1 if none
    int halfmove_clock;
    int fullmove_number;
    int step_count;
    int max_steps;
    float illegal_move_penalty;
    uint64_t rng_state;

    // Two-phase state per player (0=White, 1=Black)
    PhaseState phase_state[2];

    // Reward config for two-phase actions
    float reward_invalid_piece;   // default -0.01
    float reward_invalid_move;    // default -0.01
    float reward_valid_piece;     // default 0.0
    float reward_valid_move;      // default 0.0
} ChessEnv;

/* Read action[idx] respecting the actual numpy dtype (int32 or int64). */
static inline int get_action(const ChessEnv* env, int idx) {
    if (env->action_itemsize == 8)
        return (int)((int64_t*)env->actions)[idx];
    return ((int32_t*)env->actions)[idx];
}

// ============================================================================
// Pre-computed move tables
// ============================================================================

static const int KNIGHT_OFFSETS[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    { 1, -2}, { 1, 2}, { 2, -1}, { 2, 1}
};

static const int KING_OFFSETS[8][2] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1},          { 0, 1},
    { 1, -1}, { 1, 0}, { 1, 1}
};

static const int BISHOP_DIRS[4][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

static const int ROOK_DIRS[4][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

// ============================================================================
// Helper functions
// ============================================================================

static inline uint64_t chess_xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int chess_rand_int(uint64_t* state, int max) {
    if (max <= 0) return 0;
    return (int)(chess_xorshift64(state) % (uint64_t)max);
}

static inline int is_white_piece(int8_t p) {
    return p >= WP && p <= WK;
}

static inline int is_black_piece(int8_t p) {
    return p >= BP && p <= BK;
}

static inline int is_own_piece(int8_t p, int player) {
    if (player == 0) return is_white_piece(p);
    return is_black_piece(p);
}

static inline int is_enemy_piece(int8_t p, int player) {
    if (player == 0) return is_black_piece(p);
    return is_white_piece(p);
}

static inline int piece_color(int8_t p) {
    if (p >= WP && p <= WK) return 0;
    if (p >= BP && p <= BK) return 1;
    return -1;
}

static inline int sq_row(int sq) {
    return sq / 8;
}

static inline int sq_col(int sq) {
    return sq % 8;
}

static inline int make_sq(int row, int col) {
    return row * 8 + col;
}

static inline int on_board(int row, int col) {
    return row >= 0 && row < 8 && col >= 0 && col < 8;
}

// ============================================================================
// Attack detection
// ============================================================================

// Check if square sq is attacked by any piece belonging to by_player
static inline int is_square_attacked(ChessEnv* env, int sq, int by_player) {
    int r = sq_row(sq);
    int c = sq_col(sq);

    // Pawn attacks
    if (by_player == 0) {
        // White pawns attack upward-diagonally
        if (r > 0) {
            if (c > 0 && env->board[make_sq(r - 1, c - 1)] == WP) return 1;
            if (c < 7 && env->board[make_sq(r - 1, c + 1)] == WP) return 1;
        }
    } else {
        // Black pawns attack downward-diagonally
        if (r < 7) {
            if (c > 0 && env->board[make_sq(r + 1, c - 1)] == BP) return 1;
            if (c < 7 && env->board[make_sq(r + 1, c + 1)] == BP) return 1;
        }
    }

    // Knight attacks
    int8_t enemy_knight = (by_player == 0) ? WN : BN;
    for (int i = 0; i < 8; i++) {
        int nr = r + KNIGHT_OFFSETS[i][0];
        int nc = c + KNIGHT_OFFSETS[i][1];
        if (on_board(nr, nc) && env->board[make_sq(nr, nc)] == enemy_knight) {
            return 1;
        }
    }

    // King attacks
    int8_t enemy_king = (by_player == 0) ? WK : BK;
    for (int i = 0; i < 8; i++) {
        int nr = r + KING_OFFSETS[i][0];
        int nc = c + KING_OFFSETS[i][1];
        if (on_board(nr, nc) && env->board[make_sq(nr, nc)] == enemy_king) {
            return 1;
        }
    }

    // Bishop/Queen diagonal attacks
    int8_t enemy_bishop = (by_player == 0) ? WB : BB;
    int8_t enemy_queen = (by_player == 0) ? WQ : BQ;
    for (int d = 0; d < 4; d++) {
        int dr = BISHOP_DIRS[d][0];
        int dc = BISHOP_DIRS[d][1];
        int nr = r + dr;
        int nc = c + dc;
        while (on_board(nr, nc)) {
            int8_t p = env->board[make_sq(nr, nc)];
            if (p != EMPTY) {
                if (p == enemy_bishop || p == enemy_queen) return 1;
                break;
            }
            nr += dr;
            nc += dc;
        }
    }

    // Rook/Queen straight attacks
    int8_t enemy_rook = (by_player == 0) ? WR : BR;
    for (int d = 0; d < 4; d++) {
        int dr = ROOK_DIRS[d][0];
        int dc = ROOK_DIRS[d][1];
        int nr = r + dr;
        int nc = c + dc;
        while (on_board(nr, nc)) {
            int8_t p = env->board[make_sq(nr, nc)];
            if (p != EMPTY) {
                if (p == enemy_rook || p == enemy_queen) return 1;
                break;
            }
            nr += dr;
            nc += dc;
        }
    }

    return 0;
}

// Find king square for given player
static inline int find_king(ChessEnv* env, int player) {
    int8_t king = (player == 0) ? WK : BK;
    for (int sq = 0; sq < 64; sq++) {
        if (env->board[sq] == king) return sq;
    }
    return -1; // Should never happen in a valid position
}

// Check if player's king is in check
static inline int is_in_check(ChessEnv* env, int player) {
    int king_sq = find_king(env, player);
    if (king_sq < 0) return 0;
    int opponent = 1 - player;
    return is_square_attacked(env, king_sq, opponent);
}

// ============================================================================
// Move generation (pseudo-legal then filtered for legality)
// ============================================================================

// Try adding a move; returns 1 if added, 0 if buffer full
static inline int add_move(int moves[], int* count, int max_moves, int from_sq, int to_sq) {
    if (*count >= max_moves) return 0;
    moves[(*count)++] = from_sq * 64 + to_sq;
    return 1;
}

// Generate all pseudo-legal moves for current player
static int generate_pseudo_legal_moves(ChessEnv* env, int moves[], int max_moves) {
    int count = 0;
    int player = env->current_player;

    for (int sq = 0; sq < 64; sq++) {
        int8_t piece = env->board[sq];
        if (!is_own_piece(piece, player)) continue;

        int r = sq_row(sq);
        int c = sq_col(sq);

        // Determine piece type (normalize to white piece type for logic)
        int ptype = piece;
        if (ptype >= BP) ptype -= 6; // BP->WP, BN->WN, etc.

        switch (ptype) {
        case WP: {
            int dir = (player == 0) ? 1 : -1;
            int start_row = (player == 0) ? 1 : 6;
            // Single push
            int nr = r + dir;
            if (nr >= 0 && nr < 8) {
                int target = make_sq(nr, c);
                if (env->board[target] == EMPTY) {
                    add_move(moves, &count, max_moves, sq, target);

                    // Double push from starting row
                    if (r == start_row) {
                        int nr2 = r + 2 * dir;
                        int target2 = make_sq(nr2, c);
                        if (env->board[target2] == EMPTY) {
                            add_move(moves, &count, max_moves, sq, target2);
                        }
                    }
                }

                // Diagonal captures
                for (int dc = -1; dc <= 1; dc += 2) {
                    int nc = c + dc;
                    if (nc >= 0 && nc < 8) {
                        int cap_sq = make_sq(nr, nc);
                        if (is_enemy_piece(env->board[cap_sq], player)) {
                            add_move(moves, &count, max_moves, sq, cap_sq);
                        }
                        // En passant
                        if (cap_sq == env->en_passant_square) {
                            add_move(moves, &count, max_moves, sq, cap_sq);
                        }
                    }
                }
            }
            break;
        }

        case WN: {
            for (int i = 0; i < 8; i++) {
                int nr = r + KNIGHT_OFFSETS[i][0];
                int nc = c + KNIGHT_OFFSETS[i][1];
                if (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    if (!is_own_piece(env->board[target], player)) {
                        add_move(moves, &count, max_moves, sq, target);
                    }
                }
            }
            break;
        }

        case WB: {
            for (int d = 0; d < 4; d++) {
                int dr = BISHOP_DIRS[d][0];
                int dc = BISHOP_DIRS[d][1];
                int nr = r + dr;
                int nc = c + dc;
                while (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    int8_t tp = env->board[target];
                    if (is_own_piece(tp, player)) break;
                    add_move(moves, &count, max_moves, sq, target);
                    if (tp != EMPTY) break; // Captured enemy
                    nr += dr;
                    nc += dc;
                }
            }
            break;
        }

        case WR: {
            for (int d = 0; d < 4; d++) {
                int dr = ROOK_DIRS[d][0];
                int dc = ROOK_DIRS[d][1];
                int nr = r + dr;
                int nc = c + dc;
                while (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    int8_t tp = env->board[target];
                    if (is_own_piece(tp, player)) break;
                    add_move(moves, &count, max_moves, sq, target);
                    if (tp != EMPTY) break;
                    nr += dr;
                    nc += dc;
                }
            }
            break;
        }

        case WQ: {
            // Queen = bishop + rook moves
            for (int d = 0; d < 4; d++) {
                int dr = BISHOP_DIRS[d][0];
                int dc = BISHOP_DIRS[d][1];
                int nr = r + dr;
                int nc = c + dc;
                while (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    int8_t tp = env->board[target];
                    if (is_own_piece(tp, player)) break;
                    add_move(moves, &count, max_moves, sq, target);
                    if (tp != EMPTY) break;
                    nr += dr;
                    nc += dc;
                }
            }
            for (int d = 0; d < 4; d++) {
                int dr = ROOK_DIRS[d][0];
                int dc = ROOK_DIRS[d][1];
                int nr = r + dr;
                int nc = c + dc;
                while (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    int8_t tp = env->board[target];
                    if (is_own_piece(tp, player)) break;
                    add_move(moves, &count, max_moves, sq, target);
                    if (tp != EMPTY) break;
                    nr += dr;
                    nc += dc;
                }
            }
            break;
        }

        case WK: {
            // Normal king moves
            for (int i = 0; i < 8; i++) {
                int nr = r + KING_OFFSETS[i][0];
                int nc = c + KING_OFFSETS[i][1];
                if (on_board(nr, nc)) {
                    int target = make_sq(nr, nc);
                    if (!is_own_piece(env->board[target], player)) {
                        add_move(moves, &count, max_moves, sq, target);
                    }
                }
            }

            // Castling
            int opponent = 1 - player;
            if (player == 0) {
                // White kingside: e1(4) -> g1(6), rook h1(7) -> f1(5)
                if ((env->castling_rights & CASTLE_WK) &&
                    env->board[5] == EMPTY && env->board[6] == EMPTY &&
                    !is_square_attacked(env, 4, opponent) &&
                    !is_square_attacked(env, 5, opponent) &&
                    !is_square_attacked(env, 6, opponent)) {
                    add_move(moves, &count, max_moves, 4, 6);
                }
                // White queenside: e1(4) -> c1(2), rook a1(0) -> d1(3)
                if ((env->castling_rights & CASTLE_WQ) &&
                    env->board[3] == EMPTY && env->board[2] == EMPTY && env->board[1] == EMPTY &&
                    !is_square_attacked(env, 4, opponent) &&
                    !is_square_attacked(env, 3, opponent) &&
                    !is_square_attacked(env, 2, opponent)) {
                    add_move(moves, &count, max_moves, 4, 2);
                }
            } else {
                // Black kingside: e8(60) -> g8(62), rook h8(63) -> f8(61)
                if ((env->castling_rights & CASTLE_BK) &&
                    env->board[61] == EMPTY && env->board[62] == EMPTY &&
                    !is_square_attacked(env, 60, opponent) &&
                    !is_square_attacked(env, 61, opponent) &&
                    !is_square_attacked(env, 62, opponent)) {
                    add_move(moves, &count, max_moves, 60, 62);
                }
                // Black queenside: e8(60) -> c8(58), rook a8(56) -> d8(59)
                if ((env->castling_rights & CASTLE_BQ) &&
                    env->board[59] == EMPTY && env->board[58] == EMPTY && env->board[57] == EMPTY &&
                    !is_square_attacked(env, 60, opponent) &&
                    !is_square_attacked(env, 59, opponent) &&
                    !is_square_attacked(env, 58, opponent)) {
                    add_move(moves, &count, max_moves, 60, 58);
                }
            }
            break;
        }
        } // switch
    }

    return count;
}

// Test if a pseudo-legal move is legal (doesn't leave own king in check)
static inline int is_move_legal(ChessEnv* env, int from_sq, int to_sq) {
    int player = env->current_player;

    // Save state
    int8_t captured = env->board[to_sq];
    int8_t moved = env->board[from_sq];
    int8_t ep_captured = EMPTY;
    int ep_capture_sq = -1;

    // Handle en passant capture for legality check
    int ptype = moved;
    if (ptype >= BP) ptype -= 6;
    if (ptype == WP && to_sq == env->en_passant_square) {
        int dir = (player == 0) ? -1 : 1;
        ep_capture_sq = to_sq + dir * 8;
        ep_captured = env->board[ep_capture_sq];
        env->board[ep_capture_sq] = EMPTY;
    }

    // Make the move temporarily
    env->board[to_sq] = moved;
    env->board[from_sq] = EMPTY;

    // Handle castling king move - need to also move the rook temporarily
    int rook_from = -1, rook_to = -1;
    int8_t rook_piece = EMPTY;
    if (ptype == WK) {
        if (from_sq == 4 && to_sq == 6) {   // White kingside
            rook_from = 7; rook_to = 5; rook_piece = WR;
        } else if (from_sq == 4 && to_sq == 2) {   // White queenside
            rook_from = 0; rook_to = 3; rook_piece = WR;
        } else if (from_sq == 60 && to_sq == 62) {  // Black kingside
            rook_from = 63; rook_to = 61; rook_piece = BR;
        } else if (from_sq == 60 && to_sq == 58) {  // Black queenside
            rook_from = 56; rook_to = 59; rook_piece = BR;
        }
        if (rook_from >= 0) {
            env->board[rook_to] = rook_piece;
            env->board[rook_from] = EMPTY;
        }
    }

    int in_check = is_in_check(env, player);

    // Undo the move
    env->board[from_sq] = moved;
    env->board[to_sq] = captured;
    if (ep_capture_sq >= 0) {
        env->board[ep_capture_sq] = ep_captured;
    }
    if (rook_from >= 0) {
        env->board[rook_from] = rook_piece;
        env->board[rook_to] = EMPTY;
    }

    return !in_check;
}

// Generate all legal moves for current player
static int generate_legal_moves(ChessEnv* env, int moves[], int max_moves) {
    int pseudo_moves[CHESS_MAX_MOVES];
    int pseudo_count = generate_pseudo_legal_moves(env, pseudo_moves, CHESS_MAX_MOVES);

    int legal_count = 0;
    for (int i = 0; i < pseudo_count; i++) {
        int from_sq = pseudo_moves[i] / 64;
        int to_sq = pseudo_moves[i] % 64;
        if (is_move_legal(env, from_sq, to_sq)) {
            if (legal_count < max_moves) {
                moves[legal_count++] = pseudo_moves[i];
            }
        }
    }

    return legal_count;
}

// Early-exit: returns 1 as soon as any legal move is found, 0 if none exist.
static int has_any_legal_move(ChessEnv* env) {
    int pseudo_moves[CHESS_MAX_MOVES];
    int pseudo_count = generate_pseudo_legal_moves(env, pseudo_moves, CHESS_MAX_MOVES);

    for (int i = 0; i < pseudo_count; i++) {
        int from_sq = pseudo_moves[i] / 64;
        int to_sq = pseudo_moves[i] % 64;
        if (is_move_legal(env, from_sq, to_sq)) {
            return 1;
        }
    }
    return 0;
}

// ============================================================================
// Move application
// ============================================================================

// Apply a move with explicit promotion piece type.
// promo_piece: the actual piece to place (e.g. WQ, WR, WB, WN, or BQ etc.)
//              Pass EMPTY (0) for auto-queen or non-promotion moves.
static void apply_move_ex(ChessEnv* env, int from_sq, int to_sq, int8_t promo_piece) {
    int8_t piece = env->board[from_sq];
    int8_t captured = env->board[to_sq];
    int player = env->current_player;

    int ptype = piece;
    if (ptype >= BP) ptype -= 6;

    int is_capture = (captured != EMPTY);

    // Handle en passant capture
    if (ptype == WP && to_sq == env->en_passant_square) {
        int dir = (player == 0) ? -1 : 1;
        int ep_cap_sq = to_sq + dir * 8;
        env->board[ep_cap_sq] = EMPTY;
        is_capture = 1;
    }

    // Move piece
    env->board[to_sq] = piece;
    env->board[from_sq] = EMPTY;

    // Pawn promotion
    if (ptype == WP) {
        int promo_row = (player == 0) ? 7 : 0;
        if (sq_row(to_sq) == promo_row) {
            if (promo_piece != EMPTY) {
                env->board[to_sq] = promo_piece;
            } else {
                // Auto-queen fallback
                env->board[to_sq] = (player == 0) ? WQ : BQ;
            }
        }
    }

    // Handle castling rook movement
    if (ptype == WK) {
        if (from_sq == 4 && to_sq == 6) {       // White kingside
            env->board[5] = WR;
            env->board[7] = EMPTY;
        } else if (from_sq == 4 && to_sq == 2) { // White queenside
            env->board[3] = WR;
            env->board[0] = EMPTY;
        } else if (from_sq == 60 && to_sq == 62) { // Black kingside
            env->board[61] = BR;
            env->board[63] = EMPTY;
        } else if (from_sq == 60 && to_sq == 58) { // Black queenside
            env->board[59] = BR;
            env->board[56] = EMPTY;
        }
    }

    // Update castling rights
    if (piece == WK) {
        env->castling_rights &= ~(CASTLE_WK | CASTLE_WQ);
    } else if (piece == BK) {
        env->castling_rights &= ~(CASTLE_BK | CASTLE_BQ);
    }
    // Rook moves or gets captured
    if (from_sq == 0 || to_sq == 0)   env->castling_rights &= ~CASTLE_WQ;
    if (from_sq == 7 || to_sq == 7)   env->castling_rights &= ~CASTLE_WK;
    if (from_sq == 56 || to_sq == 56) env->castling_rights &= ~CASTLE_BQ;
    if (from_sq == 63 || to_sq == 63) env->castling_rights &= ~CASTLE_BK;

    // Update en passant square
    env->en_passant_square = -1;
    if (ptype == WP) {
        int from_row = sq_row(from_sq);
        int to_row = sq_row(to_sq);
        if ((to_row - from_row == 2) || (from_row - to_row == 2)) {
            // Double pawn push - en passant target is the square behind
            env->en_passant_square = (from_sq + to_sq) / 2;
        }
    }

    // Update halfmove clock
    if (ptype == WP || is_capture) {
        env->halfmove_clock = 0;
    } else {
        env->halfmove_clock++;
    }

    // Update fullmove number after Black's move
    if (player == 1) {
        env->fullmove_number++;
    }
}

// Legacy wrapper: auto-queen promotion
static void apply_move(ChessEnv* env, int from_sq, int to_sq) {
    apply_move_ex(env, from_sq, to_sq, EMPTY);
}

// ============================================================================
// Game end detection
// ============================================================================

static int check_game_end(ChessEnv* env, int has_legal) {
    // Fifty-move rule
    if (env->halfmove_clock >= 100) {
        return GAME_FIFTY_MOVE;
    }

    // has_legal: 1 if current player has at least one legal move, 0 if none
    if (!has_legal) {
        if (is_in_check(env, env->current_player)) {
            return GAME_CHECKMATE;
        }
        return GAME_STALEMATE;
    }

    // Insufficient material
    // Count pieces
    int white_knights = 0, white_bishops = 0, white_others = 0;
    int black_knights = 0, black_bishops = 0, black_others = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t p = env->board[sq];
        switch (p) {
            case WP: case WR: case WQ: white_others++; break;
            case WN: white_knights++; break;
            case WB: white_bishops++; break;
            case BP: case BR: case BQ: black_others++; break;
            case BN: black_knights++; break;
            case BB: black_bishops++; break;
            default: break;
        }
    }

    if (white_others == 0 && black_others == 0) {
        int wminor = white_knights + white_bishops;
        int bminor = black_knights + black_bishops;
        // K vs K
        if (wminor == 0 && bminor == 0) return GAME_INSUFFICIENT;
        // K+B vs K or K+N vs K
        if (wminor <= 1 && bminor == 0) return GAME_INSUFFICIENT;
        if (bminor <= 1 && wminor == 0) return GAME_INSUFFICIENT;
    }

    return GAME_ONGOING;
}

// ============================================================================
// Two-phase action processing
// ============================================================================

// Check if a move from from_sq to to_sq is a promotion move
static inline int is_promotion_move(ChessEnv* env, int from_sq, int to_sq) {
    int8_t piece = env->board[from_sq];
    int ptype = piece;
    if (ptype >= BP) ptype -= 6;
    if (ptype != WP) return 0;
    int player = env->current_player;
    int promo_row = (player == 0) ? 7 : 0;
    return sq_row(to_sq) == promo_row;
}

// Flip a square for Black's perspective (row flip, col stays)
static inline int flip_sq(int sq) {
    int r = sq_row(sq);
    int c = sq_col(sq);
    return make_sq(7 - r, c);
}

// Process a player's action in the two-phase system.
// Returns 1 if a chess move was completed (board changed, turn should switch).
// Returns 0 if still in sub-step (phase change or pass).
static int process_player_action(ChessEnv* env, int action, int player) {
    PhaseState* ps = &env->phase_state[player];

    // PASS action
    if (action == CHESS_PASS_ACTION) {
        // Pass is valid when it's NOT this player's turn
        if (env->current_player != player) {
            return 0; // Valid pass, no move
        }
        // Invalid pass (it IS our turn) - penalize
        env->rewards[player] += env->reward_invalid_move;
        env->log.illegal_moves += 1.0f;
        return 0;
    }

    // If it's not our turn, any non-pass action is invalid
    if (env->current_player != player) {
        env->rewards[player] += env->reward_invalid_move;
        env->log.illegal_moves += 1.0f;
        return 0;
    }

    if (ps->pick_phase == 0) {
        // Phase 0: Pick a piece
        if (action < 0 || action > 63) {
            env->rewards[player] += env->reward_invalid_piece;
            env->log.illegal_moves += 1.0f;
            return 0;
        }

        // Convert from player's perspective to absolute square
        int abs_sq = (player == 0) ? action : flip_sq(action);

        // Check if this square has our piece with legal moves
        if (!is_own_piece(env->board[abs_sq], player)) {
            env->rewards[player] += env->reward_invalid_piece;
            env->log.illegal_moves += 1.0f;
            return 0;
        }

        // Generate all legal moves and filter for this piece
        int all_legal[CHESS_MAX_MOVES];
        int num_legal = generate_legal_moves(env, all_legal, CHESS_MAX_MOVES);

        ps->valid_dest_count = 0;
        for (int i = 0; i < num_legal; i++) {
            int from = all_legal[i] / 64;
            if (from == abs_sq) {
                ps->valid_dest_moves[ps->valid_dest_count++] = all_legal[i];
            }
        }

        if (ps->valid_dest_count == 0) {
            // Piece has no legal moves
            env->rewards[player] += env->reward_invalid_piece;
            env->log.illegal_moves += 1.0f;
            return 0;
        }

        // Valid piece selection - transition to phase 1
        ps->selected_square = abs_sq;
        ps->pick_phase = 1;
        env->rewards[player] += env->reward_valid_piece;
        return 0; // No chess move yet, just phase transition

    } else {
        // Phase 1: Pick destination or promotion
        int from_sq = ps->selected_square;
        int to_sq = -1;
        int8_t promo_piece = EMPTY;

        if (action >= 0 && action <= 63) {
            // Destination square - convert from player perspective to absolute
            to_sq = (player == 0) ? action : flip_sq(action);
        } else if (action >= 64 && action <= 95) {
            // Promotion: action = 64 + type*8 + file
            // type: 0=Queen, 1=Rook, 2=Bishop, 3=Knight
            int promo_idx = action - 64;
            int promo_type = promo_idx / 8;  // 0-3
            int promo_file = promo_idx % 8;  // 0-7

            // The destination row for promotion
            int promo_row = (player == 0) ? 7 : 0;
            to_sq = make_sq(promo_row, promo_file);

            // Map promo_type to piece
            int8_t promo_types_white[4] = {WQ, WR, WB, WN};
            int8_t promo_types_black[4] = {BQ, BR, BB, BN};
            promo_piece = (player == 0) ? promo_types_white[promo_type] : promo_types_black[promo_type];
        } else {
            // Invalid action range in phase 1
            ps->pick_phase = 0;
            ps->selected_square = -1;
            env->rewards[player] += env->reward_invalid_move;
            env->log.illegal_moves += 1.0f;
            return 0;
        }

        // Check if from_sq -> to_sq is in valid_dest_moves
        int move_encoded = from_sq * 64 + to_sq;
        int found = 0;
        for (int i = 0; i < ps->valid_dest_count; i++) {
            if (ps->valid_dest_moves[i] == move_encoded) {
                found = 1;
                break;
            }
        }

        if (!found) {
            // Invalid destination - reset to phase 0
            ps->pick_phase = 0;
            ps->selected_square = -1;
            env->rewards[player] += env->reward_invalid_move;
            env->log.illegal_moves += 1.0f;
            return 0;
        }

        // Check if this is a promotion move but action was non-promotion (0-63)
        // In that case, auto-queen (promo_piece stays EMPTY -> apply_move_ex auto-queens)
        // This allows the simpler destination-only path for promotions too.

        // Valid move - apply it
        apply_move_ex(env, from_sq, to_sq, promo_piece);

        // Reset phase state
        ps->pick_phase = 0;
        ps->selected_square = -1;
        env->rewards[player] += env->reward_valid_move;

        return 1; // Chess move completed
    }
}

// ============================================================================
// Observation writing
// ============================================================================

// Helper: compute valid_pieces mask for a given player
// Sets mask[sq]=255 for squares (in player's perspective) that have pieces with legal moves
static void compute_valid_pieces_mask(ChessEnv* env, int player, unsigned char* mask) {
    memset(mask, 0, 64);

    if (env->current_player != player) return; // Not our turn

    int all_legal[CHESS_MAX_MOVES];
    int num_legal = generate_legal_moves(env, all_legal, CHESS_MAX_MOVES);

    // Track which from-squares appear in legal moves
    unsigned char has_legal_from[64];
    memset(has_legal_from, 0, 64);
    for (int i = 0; i < num_legal; i++) {
        int from = all_legal[i] / 64;
        has_legal_from[from] = 1;
    }

    // Convert to player's perspective
    for (int sq = 0; sq < 64; sq++) {
        if (has_legal_from[sq]) {
            int psq = (player == 0) ? sq : flip_sq(sq);
            mask[psq] = 255;
        }
    }
}

// Helper: compute valid_dests mask for a player in phase 1
// Sets mask[sq]=255 for valid destination squares (in player's perspective)
static void compute_valid_dests_mask(ChessEnv* env, int player, unsigned char* dest_mask, unsigned char* promo_mask) {
    memset(dest_mask, 0, 64);
    memset(promo_mask, 0, 32);

    PhaseState* ps = &env->phase_state[player];
    if (ps->pick_phase != 1) return;

    int from_sq = ps->selected_square;

    for (int i = 0; i < ps->valid_dest_count; i++) {
        int to = ps->valid_dest_moves[i] % 64;
        int psq = (player == 0) ? to : flip_sq(to);

        // Check if this is a promotion move
        if (is_promotion_move(env, from_sq, to)) {
            // For promotion moves, mark the promotion actions
            int file = sq_col(to);
            // Mark all 4 promotion types for this file
            for (int pt = 0; pt < 4; pt++) {
                promo_mask[pt * 8 + file] = 255;
            }
            // Also mark the destination square (for auto-queen via action 0-63)
            dest_mask[psq] = 255;
        } else {
            dest_mask[psq] = 255;
        }
    }
}

static void write_observations(ChessEnv* env) {
    unsigned char* white_obs = env->observations;
    unsigned char* black_obs = env->observations + env->obs_stride;

    // === Board (offset 0, size 64) ===

    // Write board for White (agent 0): as-is
    for (int sq = 0; sq < 64; sq++) {
        white_obs[sq] = (unsigned char)env->board[sq];
    }

    // Write board for Black (agent 1): flipped board, swapped colors
    for (int sq = 0; sq < 64; sq++) {
        int r = sq_row(sq);
        int c = sq_col(sq);
        int flipped_sq = make_sq(7 - r, c);
        int8_t p = env->board[flipped_sq];
        unsigned char obs_val;
        if (p == EMPTY) {
            obs_val = 0;
        } else if (is_white_piece(p)) {
            obs_val = (unsigned char)(p + 6);
        } else {
            obs_val = (unsigned char)(p - 6);
        }
        black_obs[sq] = obs_val;
    }

    // === Side to move one-hot (offset 64, size 2) ===
    white_obs[64] = (env->current_player == 0) ? 255 : 0;
    white_obs[65] = (env->current_player == 0) ? 0 : 255;
    black_obs[64] = (env->current_player == 1) ? 255 : 0;
    black_obs[65] = (env->current_player == 1) ? 0 : 255;

    // === Castling rights (offset 66, size 4): own_K, own_Q, opp_K, opp_Q ===
    white_obs[66] = (env->castling_rights & CASTLE_WK) ? 255 : 0;
    white_obs[67] = (env->castling_rights & CASTLE_WQ) ? 255 : 0;
    white_obs[68] = (env->castling_rights & CASTLE_BK) ? 255 : 0;
    white_obs[69] = (env->castling_rights & CASTLE_BQ) ? 255 : 0;

    black_obs[66] = (env->castling_rights & CASTLE_BK) ? 255 : 0;
    black_obs[67] = (env->castling_rights & CASTLE_BQ) ? 255 : 0;
    black_obs[68] = (env->castling_rights & CASTLE_WK) ? 255 : 0;
    black_obs[69] = (env->castling_rights & CASTLE_WQ) ? 255 : 0;

    // === En passant file (offset 70, size 1) ===
    if (env->en_passant_square >= 0 && env->current_player == 0) {
        white_obs[70] = (unsigned char)sq_col(env->en_passant_square);
    } else {
        white_obs[70] = 255; // None
    }
    if (env->en_passant_square >= 0 && env->current_player == 1) {
        black_obs[70] = (unsigned char)sq_col(env->en_passant_square);
    } else {
        black_obs[70] = 255; // None
    }

    // === Phase one-hot (offset 71, size 2) ===
    white_obs[71] = (env->phase_state[0].pick_phase == 0) ? 255 : 0;
    white_obs[72] = (env->phase_state[0].pick_phase == 1) ? 255 : 0;
    black_obs[71] = (env->phase_state[1].pick_phase == 0) ? 255 : 0;
    black_obs[72] = (env->phase_state[1].pick_phase == 1) ? 255 : 0;

    // === Selected piece plane (offset 73, size 64) ===
    memset(white_obs + 73, 0, 64);
    memset(black_obs + 73, 0, 64);
    if (env->phase_state[0].pick_phase == 1 && env->phase_state[0].selected_square >= 0) {
        int psq = env->phase_state[0].selected_square; // White: no flip needed
        white_obs[73 + psq] = 255;
    }
    if (env->phase_state[1].pick_phase == 1 && env->phase_state[1].selected_square >= 0) {
        int psq = flip_sq(env->phase_state[1].selected_square); // Black: flip
        black_obs[73 + psq] = 255;
    }

    // === Valid pieces mask (offset 137, size 64) ===
    compute_valid_pieces_mask(env, 0, white_obs + 137);
    compute_valid_pieces_mask(env, 1, black_obs + 137);

    // === Valid destinations mask (offset 201, size 64) + Valid promotions (offset 265, size 32) ===
    compute_valid_dests_mask(env, 0, white_obs + 201, white_obs + 265);
    compute_valid_dests_mask(env, 1, black_obs + 201, black_obs + 265);

    // === Self in check (offset 297, size 1) ===
    white_obs[297] = is_in_check(env, 0) ? 255 : 0;
    black_obs[297] = is_in_check(env, 1) ? 255 : 0;

    // === Opponent in check (offset 298, size 1) ===
    white_obs[298] = is_in_check(env, 1) ? 255 : 0;
    black_obs[298] = is_in_check(env, 0) ? 255 : 0;

    // === Rule50 counter (offset 299, size 1) ===
    int rule50_scaled = (env->halfmove_clock * 255) / 100;
    if (rule50_scaled > 255) rule50_scaled = 255;
    white_obs[299] = (unsigned char)rule50_scaled;
    black_obs[299] = (unsigned char)rule50_scaled;

    // === Pass valid (offset 300, size 1) ===
    // Pass is valid when it's NOT this player's turn
    white_obs[300] = (env->current_player != 0) ? 255 : 0;
    black_obs[300] = (env->current_player != 1) ? 255 : 0;
}

// ============================================================================
// Board setup
// ============================================================================

static void setup_initial_board(ChessEnv* env) {
    memset(env->board, EMPTY, 64);

    // Row 0 (rank 1): White back rank
    env->board[0] = WR;
    env->board[1] = WN;
    env->board[2] = WB;
    env->board[3] = WQ;
    env->board[4] = WK;
    env->board[5] = WB;
    env->board[6] = WN;
    env->board[7] = WR;

    // Row 1 (rank 2): White pawns
    for (int c = 0; c < 8; c++) {
        env->board[make_sq(1, c)] = WP;
    }

    // Row 6 (rank 7): Black pawns
    for (int c = 0; c < 8; c++) {
        env->board[make_sq(6, c)] = BP;
    }

    // Row 7 (rank 8): Black back rank
    env->board[56] = BR;
    env->board[57] = BN;
    env->board[58] = BB;
    env->board[59] = BQ;
    env->board[60] = BK;
    env->board[61] = BB;
    env->board[62] = BN;
    env->board[63] = BR;
}

// ============================================================================
// PufferLib interface functions
// ============================================================================

void init(ChessEnv* env) {
    env->max_steps = 256;
    env->illegal_move_penalty = -0.1f;
    env->obs_stride = CHESS_OBS_SIZE;
    env->reward_invalid_piece = -0.01f;
    env->reward_invalid_move = -0.01f;
    env->reward_valid_piece = 0.0f;
    env->reward_valid_move = 0.0f;
}

void c_reset(ChessEnv* env) {
    setup_initial_board(env);
    env->current_player = 0;
    env->castling_rights = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    env->en_passant_square = -1;
    env->halfmove_clock = 0;
    env->fullmove_number = 1;
    env->step_count = 0;

    // Reset phase state for both players
    env->phase_state[0].pick_phase = 0;
    env->phase_state[0].selected_square = -1;
    env->phase_state[0].valid_dest_count = 0;
    env->phase_state[1].pick_phase = 0;
    env->phase_state[1].selected_square = -1;
    env->phase_state[1].valid_dest_count = 0;

    // Clear rewards and terminals for both agents
    env->rewards[0] = 0.0f;
    env->rewards[1] = 0.0f;
    env->terminals[0] = 0;
    env->terminals[1] = 0;

    // Reset log
    env->log.episode_length = 0;
    env->log.episode_return = 0;
    env->log.white_wins = 0;
    env->log.black_wins = 0;
    env->log.draws = 0;
    env->log.illegal_moves = 0;
    env->log.n = 0;

    write_observations(env);
}

void c_step(ChessEnv* env) {
    // Auto-reset: if terminal at start of step, reset and return
    if (env->terminals[0] == 1) {
        c_reset(env);
        return;
    }

    // Clear rewards
    env->rewards[0] = 0.0f;
    env->rewards[1] = 0.0f;

    // Process both players' actions through the two-phase system
    int white_action = get_action(env, 0);
    int black_action = get_action(env, 1);

    // Clamp actions to valid range
    if (white_action < 0 || white_action >= CHESS_NUM_ACTIONS) white_action = CHESS_PASS_ACTION;
    if (black_action < 0 || black_action >= CHESS_NUM_ACTIONS) black_action = CHESS_PASS_ACTION;

    // Process current player first, then opponent
    int player = env->current_player;
    int opponent = 1 - player;

    int player_action = (player == 0) ? white_action : black_action;
    int opponent_action = (player == 0) ? black_action : white_action;

    // Process current player's action
    int move_made = process_player_action(env, player_action, player);

    // Process opponent's action (should be PASS if not their turn)
    process_player_action(env, opponent_action, opponent);

    int result = GAME_ONGOING;

    if (move_made) {
        // A chess move was completed - switch turn
        env->current_player = 1 - env->current_player;

        // Check game end
        result = check_game_end(env, has_any_legal_move(env));

        if (result == GAME_CHECKMATE) {
            int winner = player;
            int loser = opponent;
            env->rewards[winner] += 1.0f;
            env->rewards[loser] += -1.0f;
            env->terminals[0] = 1;
            env->terminals[1] = 1;

            env->log.episode_length = (float)env->step_count;
            env->log.episode_return = env->rewards[0];
            if (winner == 0) {
                env->log.white_wins = 1;
            } else {
                env->log.black_wins = 1;
            }
            env->log.n = 1;
        } else if (result == GAME_STALEMATE || result == GAME_FIFTY_MOVE || result == GAME_INSUFFICIENT) {
            env->terminals[0] = 1;
            env->terminals[1] = 1;

            env->log.episode_length = (float)env->step_count;
            env->log.episode_return = env->rewards[0];
            env->log.draws = 1;
            env->log.n = 1;
        }
    }

    // Write observations for both agents
    write_observations(env);

    // Increment step count and check truncation
    env->step_count++;
    if (env->step_count >= env->max_steps && env->terminals[0] == 0) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;

        env->log.episode_length = (float)env->step_count;
        env->log.episode_return = env->rewards[0];
        env->log.draws = 1;
        env->log.n = 1;
    }
}

void c_close(ChessEnv* env) {
    (void)env;
    // Nothing to free for stack-allocated data
}

#endif // CHESS_H
