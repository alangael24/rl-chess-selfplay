/*
 * chess.h - Complete chess engine for PufferLib self-play RL
 *
 * Single-header chess environment with 1 agent per game instance.
 * Each step, the agent controls whoever's turn it is (White or Black).
 * The learner_color alternates each reset for symmetric self-play.
 * Rewards are signed: positive when learner benefits, negative when opponent does.
 *
 * Two-phase action system (97 actions):
 *   Phase 0: Pick a piece (action 0-63 = board square)
 *   Phase 1: Pick destination (0-63) or promotion (64-95)
 *   Action 96: PASS (legacy, never valid in 1-agent mode)
 *
 * Board representation: array of 64 int8_t values + bitboards
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
#define BB_PIECE 9
#define BR 10
#define BQ 11
#define BK 12

// Legacy alias for compatibility (BB is used as variable name for Bitboard)
#define BB BB_PIECE

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
#define GAME_REPETITION 5

// ============================================================================
// Bitboard types and constants
// ============================================================================

typedef uint64_t Bitboard;

#define FileABB 0x0101010101010101ULL
#define FileBBB 0x0202020202020202ULL
#define FileCBB 0x0404040404040404ULL
#define FileDBB 0x0808080808080808ULL
#define FileEBB 0x1010101010101010ULL
#define FileFBB 0x2020202020202020ULL
#define FileGBB 0x4040404040404040ULL
#define FileHBB 0x8080808080808080ULL

#define Rank1BB 0x00000000000000FFULL
#define Rank2BB 0x000000000000FF00ULL
#define Rank3BB 0x0000000000FF0000ULL
#define Rank4BB 0x00000000FF000000ULL
#define Rank5BB 0x000000FF00000000ULL
#define Rank6BB 0x0000FF0000000000ULL
#define Rank7BB 0x00FF000000000000ULL
#define Rank8BB 0xFF00000000000000ULL

// Bitboard piece type indices (1-based, 0 = all)
#define BB_PAWN   1
#define BB_KNIGHT 2
#define BB_BISHOP 3
#define BB_ROOK   4
#define BB_QUEEN  5
#define BB_KING   6

// Global lookup tables
static Bitboard SquareBB[64];
static Bitboard FileBBTable[8];
static Bitboard RankBBTable[8];
static Bitboard PawnAttacksBB[2][64];   // [color][sq]
static Bitboard KnightAttacksBB[64];
static Bitboard KingAttacksBB[64];

// Magic bitboard tables for sliding pieces
static Bitboard BishopMasks[64];
static int BishopShifts[64];
static Bitboard BishopTable[64 * 512];  // max 512 entries per square for bishop
static Bitboard* BishopAttacks[64];     // pointers into BishopTable

static Bitboard RookMasks[64];
static int RookShifts[64];
static Bitboard RookTable[64 * 4096];   // max 4096 entries per square for rook
static Bitboard* RookAttacks[64];       // pointers into RookTable

// Hardcoded magic numbers (from well-known chess programming resources)
static const uint64_t BishopMagics[64] = {
    0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL, 0x0004040080000000ULL,
    0x0001104000000000ULL, 0x0000821040000000ULL, 0x0000410410400000ULL, 0x0000104104104000ULL,
    0x0000040404040400ULL, 0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
    0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL, 0x0000002082082000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800802004000ULL,
    0x0000800400A00000ULL, 0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
    0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL, 0x0000404004010200ULL,
    0x0000840000802000ULL, 0x0000404002011000ULL, 0x0000808001041000ULL, 0x0000404000820800ULL,
    0x0001041000202000ULL, 0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
    0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL, 0x0000808080010400ULL,
    0x0000820820004000ULL, 0x0000410410002000ULL, 0x0000082088001000ULL, 0x0000002011000800ULL,
    0x0000080100400400ULL, 0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
    0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL, 0x0000000020880000ULL,
    0x0000001002020000ULL, 0x0000040408020000ULL, 0x0004040404040000ULL, 0x0002020202020000ULL,
    0x0000104104104000ULL, 0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
    0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL, 0x0002020202020200ULL,
};

static const uint64_t RookMagics[64] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
    0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808010002000ULL, 0x0000808008001000ULL,
    0x0000808004000800ULL, 0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
    0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL, 0x0000080080100080ULL,
    0x0000040080080080ULL, 0x0000020080040080ULL, 0x0000010080800200ULL, 0x0000800080004100ULL,
    0x0000204000800080ULL, 0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
    0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL, 0x0000800040800100ULL,
    0x0000204000808000ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
    0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000800100020080ULL, 0x0000800041000080ULL,
    0x00FFFCDDFCED714AULL, 0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040010000200ULL,
    0x0000020008000100ULL, 0x0000010004000080ULL, 0x0000008002000040ULL, 0x0000004081020004ULL,
};

// Bishop relevant occupancy bit counts per square
static const int BishopBits[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
};

// Rook relevant occupancy bit counts per square
static const int RookBits[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
};

// Zobrist keys
typedef struct {
    uint64_t psq[13][64];     // [piece][square] - piece 0 (EMPTY) unused
    uint64_t enpassant[8];    // [file]
    uint64_t castling[16];    // [castling_rights]
    uint64_t side;            // side-to-move flip
} ZobristKeys;
static ZobristKeys zob;
static int bitboards_initialized = 0;

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
    float material_score;
    float positional_score;
    float invalid_action_rate;
    float chess_moves;
    float repetitions;
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

    // Bitboard representation (maintained in parallel with board[64])
    Bitboard bb_by_type[7];    // [0]=all, [1]=Pawn, [2]=Knight, [3]=Bishop, [4]=Rook, [5]=Queen, [6]=King
    Bitboard bb_by_color[2];   // [0]=White, [1]=Black
    Bitboard bb_occ;           // all occupied = bb_by_type[0]
    int piece_count[2][6];     // [color][piece_type_0based] 0=Pawn,1=Knight,2=Bishop,3=Rook,4=Queen,5=King

    // Zobrist hash (replaces old compute_position_hash)
    uint64_t zobrist_key;

    int current_player;         // 0=White, 1=Black
    uint8_t castling_rights;    // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
    int en_passant_square;      // -1 if none
    int halfmove_clock;
    int fullmove_number;
    int step_count;
    int max_steps;
    float episode_illegal_moves; // per-episode illegal moves counter
    float illegal_move_penalty;
    uint64_t rng_state;

    // Two-phase state per player (0=White, 1=Black)
    PhaseState phase_state[2];

    // Reward config for two-phase actions
    float reward_invalid_piece;   // default -0.01
    float reward_invalid_move;    // default -0.01
    float reward_valid_piece;     // default 0.0
    float reward_valid_move;      // default 0.0

    // Reward shaping config
    float reward_capture_bonus;   // default 0.0
    float reward_check_bonus;     // default 0.0
    float reward_repetition;      // default 0.0 - penalty per repeated position
    float reward_material;        // default 0.0 - scale for material delta per move
    float reward_position;        // default 0.0 - scale for positional delta per move
    float reward_castling;        // default 0.0 - one-time castling bonus
    float reward_draw;            // default 0.0 - reward for draw outcomes
    int enable_50_move_rule;      // 1=enabled, 0=disabled (default 1)
    int enable_threefold_repetition; // 1=enabled, 0=disabled (default 1)

    // Position history for threefold repetition
    uint64_t position_history[512];
    int position_history_count;

    // Per-episode chess move counter
    int episode_chess_moves;

    // FEN curriculum
    int use_curriculum;

    // SEE/hanging reward shaping
    float reward_see_hanging;         // default 0.0 - penalty for hanging pieces
    int last_see_value;               // SEE value of last move (for diagnostics)

    // 1-agent topology: learner_color alternates each reset
    int learner_color;                // 0=White, 1=Black — the "learner" perspective

    // Legal move cache
    int legal_moves_cache[256];       // cached legal moves array
    int legal_moves_cache_count;      // -1 = invalid
    uint64_t legal_moves_key;         // position hash when cached
    int legal_moves_side;             // current_player when cached
} ChessEnv;

/* Read action[idx] respecting the actual numpy dtype (int32 or int64). */
static inline int get_action(const ChessEnv* env, int idx) {
    if (env->action_itemsize == 8)
        return (int)((int64_t*)env->actions)[idx];
    return ((int32_t*)env->actions)[idx];
}

// ============================================================================
// Pre-computed move tables (kept for SEE which still uses board[64] loops)
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
// Bitboard helpers
// ============================================================================

static inline Bitboard sq_bb(int sq) { return SquareBB[sq]; }
static inline int bb_popcount(Bitboard b) { return __builtin_popcountll(b); }
static inline int bb_lsb(Bitboard b) { return __builtin_ctzll(b); }
static inline int bb_pop_lsb(Bitboard* b) {
    int sq = bb_lsb(*b);
    *b &= *b - 1;
    return sq;
}

// Piece type index (0-based): WP=0,WN=1,...,WK=5, BP=0,BN=1,...,BK=5
static inline int piece_type_idx(int8_t piece) {
    if (piece >= BP) return piece - BP;  // 7->0, 8->1, 9->2, 10->3, 11->4, 12->5
    return piece - WP;                   // 1->0, 2->1, 3->2, 4->3, 5->4, 6->5
}

static inline void bb_add_piece(ChessEnv* env, int sq, int8_t piece) {
    int color = piece_color(piece);
    int pt = piece_type_idx(piece) + 1;  // 1-based for bb_by_type
    Bitboard sqbb = sq_bb(sq);
    env->bb_by_type[0] |= sqbb;
    env->bb_by_type[pt] |= sqbb;
    env->bb_by_color[color] |= sqbb;
    env->bb_occ |= sqbb;
    env->piece_count[color][pt - 1]++;
    env->zobrist_key ^= zob.psq[(int)piece][sq];
}

static inline void bb_remove_piece(ChessEnv* env, int sq, int8_t piece) {
    int color = piece_color(piece);
    int pt = piece_type_idx(piece) + 1;
    Bitboard sqbb = sq_bb(sq);
    env->bb_by_type[0] &= ~sqbb;
    env->bb_by_type[pt] &= ~sqbb;
    env->bb_by_color[color] &= ~sqbb;
    env->bb_occ &= ~sqbb;
    env->piece_count[color][pt - 1]--;
    env->zobrist_key ^= zob.psq[(int)piece][sq];
}

static inline void bb_move_piece(ChessEnv* env, int from, int to, int8_t piece) {
    int color = piece_color(piece);
    int pt = piece_type_idx(piece) + 1;
    Bitboard fromto = sq_bb(from) ^ sq_bb(to);
    env->bb_by_type[0] ^= fromto;
    env->bb_by_type[pt] ^= fromto;
    env->bb_by_color[color] ^= fromto;
    env->bb_occ ^= fromto;
    env->zobrist_key ^= zob.psq[(int)piece][from] ^ zob.psq[(int)piece][to];
}

// ============================================================================
// Magic bitboard initialization
// ============================================================================

// Slow attack functions used only during init to build magic lookup tables
static Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int r = sq_row(sq), c = sq_col(sq);
    int dr[4] = {1, 1, -1, -1};
    int dc[4] = {1, -1, 1, -1};
    for (int d = 0; d < 4; d++) {
        int nr = r + dr[d], nc = c + dc[d];
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            int s = make_sq(nr, nc);
            attacks |= (1ULL << s);
            if (occ & (1ULL << s)) break;
            nr += dr[d]; nc += dc[d];
        }
    }
    return attacks;
}

static Bitboard rook_attacks_slow(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int r = sq_row(sq), c = sq_col(sq);
    int dr[4] = {1, -1, 0, 0};
    int dc[4] = {0, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int nr = r + dr[d], nc = c + dc[d];
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            int s = make_sq(nr, nc);
            attacks |= (1ULL << s);
            if (occ & (1ULL << s)) break;
            nr += dr[d]; nc += dc[d];
        }
    }
    return attacks;
}

// Generate the nth occupancy variation from a mask
static Bitboard index_to_occupancy(int index, Bitboard mask) {
    Bitboard occ = 0;
    int bits = bb_popcount(mask);
    for (int i = 0; i < bits; i++) {
        int sq = bb_pop_lsb(&mask);
        if (index & (1 << i)) {
            occ |= (1ULL << sq);
        }
    }
    return occ;
}

static Bitboard compute_bishop_mask(int sq) {
    Bitboard mask = 0;
    int r = sq_row(sq), c = sq_col(sq);
    int dr[4] = {1, 1, -1, -1};
    int dc[4] = {1, -1, 1, -1};
    for (int d = 0; d < 4; d++) {
        int nr = r + dr[d], nc = c + dc[d];
        while (nr > 0 && nr < 7 && nc > 0 && nc < 7) {
            mask |= (1ULL << make_sq(nr, nc));
            nr += dr[d]; nc += dc[d];
        }
    }
    return mask;
}

static Bitboard compute_rook_mask(int sq) {
    Bitboard mask = 0;
    int r = sq_row(sq), c = sq_col(sq);
    for (int nr = r + 1; nr < 7; nr++) mask |= (1ULL << make_sq(nr, c));
    for (int nr = r - 1; nr > 0; nr--) mask |= (1ULL << make_sq(nr, c));
    for (int nc = c + 1; nc < 7; nc++) mask |= (1ULL << make_sq(r, nc));
    for (int nc = c - 1; nc > 0; nc--) mask |= (1ULL << make_sq(r, nc));
    return mask;
}

static void init_bishop_magics(void) {
    Bitboard* table_ptr = BishopTable;
    for (int sq = 0; sq < 64; sq++) {
        BishopMasks[sq] = compute_bishop_mask(sq);
        BishopShifts[sq] = 64 - BishopBits[sq];
        BishopAttacks[sq] = table_ptr;

        int num_entries = 1 << BishopBits[sq];
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, BishopMasks[sq]);
            int idx = (int)((occ * BishopMagics[sq]) >> BishopShifts[sq]);
            BishopAttacks[sq][idx] = bishop_attacks_slow(sq, occ);
        }
        table_ptr += num_entries;
    }
}

static void init_rook_magics(void) {
    Bitboard* table_ptr = RookTable;
    for (int sq = 0; sq < 64; sq++) {
        RookMasks[sq] = compute_rook_mask(sq);
        RookShifts[sq] = 64 - RookBits[sq];
        RookAttacks[sq] = table_ptr;

        int num_entries = 1 << RookBits[sq];
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, RookMasks[sq]);
            int idx = (int)((occ * RookMagics[sq]) >> RookShifts[sq]);
            RookAttacks[sq][idx] = rook_attacks_slow(sq, occ);
        }
        table_ptr += num_entries;
    }
}

// Fast magic bitboard attack lookups (O(1) after init)
static inline Bitboard bishop_attacks_bb(int sq, Bitboard occ) {
    occ &= BishopMasks[sq];
    return BishopAttacks[sq][(occ * BishopMagics[sq]) >> BishopShifts[sq]];
}

static inline Bitboard rook_attacks_bb(int sq, Bitboard occ) {
    occ &= RookMasks[sq];
    return RookAttacks[sq][(occ * RookMagics[sq]) >> RookShifts[sq]];
}

static inline Bitboard queen_attacks_bb(int sq, Bitboard occ) {
    return bishop_attacks_bb(sq, occ) | rook_attacks_bb(sq, occ);
}

// ============================================================================
// Bitboard initialization (called once)
// ============================================================================

static void init_bitboards(void) {
    if (bitboards_initialized) return;

    // SquareBB
    for (int i = 0; i < 64; i++) {
        SquareBB[i] = 1ULL << i;
    }

    // File and Rank tables
    Bitboard files[8] = {FileABB, FileBBB, FileCBB, FileDBB, FileEBB, FileFBB, FileGBB, FileHBB};
    Bitboard ranks[8] = {Rank1BB, Rank2BB, Rank3BB, Rank4BB, Rank5BB, Rank6BB, Rank7BB, Rank8BB};
    for (int i = 0; i < 8; i++) {
        FileBBTable[i] = files[i];
        RankBBTable[i] = ranks[i];
    }

    // Pawn attacks
    for (int sq = 0; sq < 64; sq++) {
        int r = sq_row(sq), c = sq_col(sq);
        Bitboard w = 0, b = 0;
        // White pawns attack NW and NE
        if (r < 7 && c > 0) w |= (1ULL << make_sq(r + 1, c - 1));
        if (r < 7 && c < 7) w |= (1ULL << make_sq(r + 1, c + 1));
        // Black pawns attack SW and SE
        if (r > 0 && c > 0) b |= (1ULL << make_sq(r - 1, c - 1));
        if (r > 0 && c < 7) b |= (1ULL << make_sq(r - 1, c + 1));
        PawnAttacksBB[0][sq] = w;
        PawnAttacksBB[1][sq] = b;
    }

    // Knight attacks
    for (int sq = 0; sq < 64; sq++) {
        int r = sq_row(sq), c = sq_col(sq);
        Bitboard atk = 0;
        for (int i = 0; i < 8; i++) {
            int nr = r + KNIGHT_OFFSETS[i][0];
            int nc = c + KNIGHT_OFFSETS[i][1];
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                atk |= (1ULL << make_sq(nr, nc));
            }
        }
        KnightAttacksBB[sq] = atk;
    }

    // King attacks
    for (int sq = 0; sq < 64; sq++) {
        int r = sq_row(sq), c = sq_col(sq);
        Bitboard atk = 0;
        for (int i = 0; i < 8; i++) {
            int nr = r + KING_OFFSETS[i][0];
            int nc = c + KING_OFFSETS[i][1];
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                atk |= (1ULL << make_sq(nr, nc));
            }
        }
        KingAttacksBB[sq] = atk;
    }

    // Zobrist keys via PRNG
    uint64_t rng = 1070372ULL;
    for (int p = 0; p < 13; p++) {
        for (int sq = 0; sq < 64; sq++) {
            zob.psq[p][sq] = chess_xorshift64(&rng);
        }
    }
    for (int f = 0; f < 8; f++) {
        zob.enpassant[f] = chess_xorshift64(&rng);
    }
    for (int c = 0; c < 16; c++) {
        zob.castling[c] = chess_xorshift64(&rng);
    }
    zob.side = chess_xorshift64(&rng);

    // Init magic bitboard tables
    init_bishop_magics();
    init_rook_magics();

    bitboards_initialized = 1;
}

// Sync bitboards from board[64] - called after any board setup
static void sync_bitboards_from_board(ChessEnv* env) {
    memset(env->bb_by_type, 0, sizeof(env->bb_by_type));
    memset(env->bb_by_color, 0, sizeof(env->bb_by_color));
    memset(env->piece_count, 0, sizeof(env->piece_count));
    env->bb_occ = 0;
    env->zobrist_key = 0;

    for (int sq = 0; sq < 64; sq++) {
        int8_t p = env->board[sq];
        if (p == EMPTY) continue;
        int color = piece_color(p);
        int pt = piece_type_idx(p) + 1;
        Bitboard sqbb = 1ULL << sq;
        env->bb_by_type[0] |= sqbb;
        env->bb_by_type[pt] |= sqbb;
        env->bb_by_color[color] |= sqbb;
        env->bb_occ |= sqbb;
        env->piece_count[color][pt - 1]++;
        env->zobrist_key ^= zob.psq[(int)p][sq];
    }
    // Non-piece zobrist components
    if (env->current_player == 1) env->zobrist_key ^= zob.side;
    env->zobrist_key ^= zob.castling[env->castling_rights];
    if (env->en_passant_square >= 0)
        env->zobrist_key ^= zob.enpassant[sq_col(env->en_passant_square)];
}

// ============================================================================
// Piece values and piece-square tables for reward shaping
// ============================================================================

// Material values (centipawns): P=100, N=320, B=330, R=500, Q=900, K=0
static const int PIECE_VALUES[13] = {
    0,    // EMPTY
    100,  // WP
    320,  // WN
    330,  // WB
    500,  // WR
    900,  // WQ
    0,    // WK
    100,  // BP
    320,  // BN
    330,  // BB
    500,  // BR
    900,  // BQ
    0     // BK
};

// Pawn piece-square table (from White's perspective, index = square)
static const int PAWN_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 1 (never occupied)
     5, 10, 10,-20,-20, 10, 10,  5,   // rank 2
     5, -5,-10,  0,  0,-10, -5,  5,   // rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   // rank 4
     5,  5, 10, 25, 25, 10,  5,  5,   // rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   // rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   // rank 7
     0,  0,  0,  0,  0,  0,  0,  0    // rank 8 (promoted)
};

// Knight piece-square table (from White's perspective)
static const int KNIGHT_PST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

// Compute material score for a given player (in centipawns)
static int compute_material_score(ChessEnv* env, int player) {
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t p = env->board[sq];
        if (p == EMPTY) continue;
        if (is_own_piece(p, player)) {
            score += PIECE_VALUES[(int)p];
        }
    }
    return score;
}

// Compute positional score for a given player using PST (in centipawns)
static int compute_positional_score(ChessEnv* env, int player) {
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t p = env->board[sq];
        if (p == EMPTY) continue;
        if (!is_own_piece(p, player)) continue;

        int ptype = p;
        if (ptype >= BP) ptype -= 6;

        // For Black, mirror the square vertically to use White-perspective tables
        int pst_sq = (player == 0) ? sq : make_sq(7 - sq_row(sq), sq_col(sq));

        switch (ptype) {
        case WP: score += PAWN_PST[pst_sq]; break;
        case WN: score += KNIGHT_PST[pst_sq]; break;
        default: break;
        }
    }
    return score;
}

// Get the value of a captured piece (returns 0 if EMPTY)
static inline int captured_piece_value(int8_t piece) {
    if (piece == EMPTY) return 0;
    return PIECE_VALUES[(int)piece];
}

// Forward declaration for SEE
static inline int is_square_attacked(ChessEnv* env, int sq, int by_player);

// ============================================================================
// Static Exchange Evaluation (SEE) - still uses board[64] loops
// ============================================================================

// SEE piece values: indexed by piece code (0-12)
static const int SEE_PIECE_VALUES[13] = {
    0,      // EMPTY
    100,    // WP
    320,    // WN
    330,    // WB
    500,    // WR
    900,    // WQ
    20000,  // WK
    100,    // BP
    320,    // BN
    330,    // BB
    500,    // BR
    900,    // BQ
    20000   // BK
};

// Find all pieces of `by_player` attacking `sq`, respecting occupied[] mask.
static int find_attackers_to_sq_occ(ChessEnv* env, int sq, int by_player,
                                     int8_t atk_pieces[], int atk_squares[],
                                     int max_attackers, const int occupied[64]) {
    int count = 0;
    int r = sq_row(sq);
    int c = sq_col(sq);

    // Pawn attacks
    if (by_player == 0) {
        if (r > 0) {
            int s;
            if (c > 0) { s = make_sq(r - 1, c - 1); if (occupied[s] && env->board[s] == WP && count < max_attackers) { atk_pieces[count] = WP; atk_squares[count] = s; count++; } }
            if (c < 7) { s = make_sq(r - 1, c + 1); if (occupied[s] && env->board[s] == WP && count < max_attackers) { atk_pieces[count] = WP; atk_squares[count] = s; count++; } }
        }
    } else {
        if (r < 7) {
            int s;
            if (c > 0) { s = make_sq(r + 1, c - 1); if (occupied[s] && env->board[s] == BP && count < max_attackers) { atk_pieces[count] = BP; atk_squares[count] = s; count++; } }
            if (c < 7) { s = make_sq(r + 1, c + 1); if (occupied[s] && env->board[s] == BP && count < max_attackers) { atk_pieces[count] = BP; atk_squares[count] = s; count++; } }
        }
    }

    // Knight attacks
    int8_t knight = (by_player == 0) ? WN : BN;
    for (int i = 0; i < 8; i++) {
        int nr = r + KNIGHT_OFFSETS[i][0];
        int nc = c + KNIGHT_OFFSETS[i][1];
        if (on_board(nr, nc)) {
            int s = make_sq(nr, nc);
            if (occupied[s] && env->board[s] == knight && count < max_attackers) {
                atk_pieces[count] = knight; atk_squares[count] = s; count++;
            }
        }
    }

    // Bishop/Queen diagonal attacks (respects occupied for x-ray)
    int8_t bishop = (by_player == 0) ? WB : BB_PIECE;
    int8_t queen = (by_player == 0) ? WQ : BQ;
    for (int d = 0; d < 4; d++) {
        int dr = BISHOP_DIRS[d][0];
        int dc = BISHOP_DIRS[d][1];
        int nr = r + dr;
        int nc = c + dc;
        while (on_board(nr, nc)) {
            int s = make_sq(nr, nc);
            if (occupied[s]) {
                int8_t p = env->board[s];
                if ((p == bishop || p == queen) && count < max_attackers) {
                    atk_pieces[count] = p; atk_squares[count] = s; count++;
                }
                break; // blocked by occupied piece
            }
            nr += dr;
            nc += dc;
        }
    }

    // Rook/Queen straight attacks (respects occupied for x-ray)
    int8_t rook = (by_player == 0) ? WR : BR;
    for (int d = 0; d < 4; d++) {
        int dr = ROOK_DIRS[d][0];
        int dc = ROOK_DIRS[d][1];
        int nr = r + dr;
        int nc = c + dc;
        while (on_board(nr, nc)) {
            int s = make_sq(nr, nc);
            if (occupied[s]) {
                int8_t p = env->board[s];
                if ((p == rook || p == queen) && count < max_attackers) {
                    atk_pieces[count] = p; atk_squares[count] = s; count++;
                }
                break;
            }
            nr += dr;
            nc += dc;
        }
    }

    // King attacks
    int8_t king = (by_player == 0) ? WK : BK;
    for (int i = 0; i < 8; i++) {
        int nr = r + KING_OFFSETS[i][0];
        int nc = c + KING_OFFSETS[i][1];
        if (on_board(nr, nc)) {
            int s = make_sq(nr, nc);
            if (occupied[s] && env->board[s] == king && count < max_attackers) {
                atk_pieces[count] = king; atk_squares[count] = s; count++;
            }
        }
    }

    return count;
}

// Find index of least valuable attacker in the array
static int find_least_valuable_attacker(int8_t attackers[], int count) {
    if (count == 0) return -1;
    int best = 0;
    int best_val = SEE_PIECE_VALUES[(int)attackers[0]];
    for (int i = 1; i < count; i++) {
        int val = SEE_PIECE_VALUES[(int)attackers[i]];
        if (val < best_val) {
            best_val = val;
            best = i;
        }
    }
    return best;
}

// Full SEE for a capture
static int see_capture(ChessEnv* env, int from_sq, int to_sq, int8_t moving_piece, int8_t target_piece, int player) {
    int gain[32];
    int depth = 0;
    int side = player;

    int occupied[64];
    for (int i = 0; i < 64; i++) occupied[i] = (env->board[i] != EMPTY);
    occupied[from_sq] = 0;

    gain[depth] = SEE_PIECE_VALUES[(int)target_piece];
    int8_t current_piece = moving_piece;

    for (depth = 1; depth < 32; depth++) {
        side = 1 - side;
        gain[depth] = SEE_PIECE_VALUES[(int)current_piece] - gain[depth - 1];

        if (gain[depth] < 0 && gain[depth - 1] < 0) break;

        int8_t atk_pieces[32];
        int atk_sqs[32];
        int atk_count = find_attackers_to_sq_occ(env, to_sq, side,
                                                  atk_pieces, atk_sqs, 32, occupied);
        if (atk_count == 0) break;

        int lva = find_least_valuable_attacker(atk_pieces, atk_count);
        current_piece = atk_pieces[lva];
        occupied[atk_sqs[lva]] = 0;
    }

    for (int d = depth - 1; d > 0; d--) {
        if (-gain[d] < gain[d - 1])
            gain[d - 1] = -gain[d];
    }

    return gain[0];
}

// SEE for a quiet move to a square
static int see_square(ChessEnv* env, int from_sq, int to_sq, int8_t moving_piece, int player) {
    int opponent = 1 - player;

    if (!is_square_attacked(env, to_sq, opponent)) {
        return 0;
    }

    int occupied[64];
    for (int i = 0; i < 64; i++) occupied[i] = (env->board[i] != EMPTY);
    occupied[from_sq] = 0;
    occupied[to_sq] = 1;

    int8_t atk_pieces[32];
    int atk_sqs[32];
    int atk_count = find_attackers_to_sq_occ(env, to_sq, opponent,
                                              atk_pieces, atk_sqs, 32, occupied);
    if (atk_count == 0) return 0;

    int lva = find_least_valuable_attacker(atk_pieces, atk_count);
    int8_t opp_attacker = atk_pieces[lva];
    int opp_from = atk_sqs[lva];

    int opp_see = see_capture(env, opp_from, to_sq, opp_attacker, moving_piece, opponent);

    if (opp_see > 0) {
        return -opp_see;
    }
    return 0;
}

// ============================================================================
// Position hashing and repetition detection (now uses Zobrist)
// ============================================================================

// Returns the incremental Zobrist key
static uint64_t compute_position_hash(ChessEnv* env) {
    return env->zobrist_key;
}

// Record current position hash in history
static void record_position_hash(ChessEnv* env) {
    uint64_t hash = env->zobrist_key;
    if (env->position_history_count < 512) {
        env->position_history[env->position_history_count++] = hash;
    }
}

// Count occurrences of current position in history
static int count_position_occurrences(ChessEnv* env) {
    uint64_t current = env->zobrist_key;
    int count = 0;
    for (int i = 0; i < env->position_history_count; i++) {
        if (env->position_history[i] == current) {
            count++;
        }
    }
    return count;
}

// Check for threefold repetition (3 or more occurrences)
static int check_threefold_repetition(ChessEnv* env) {
    return count_position_occurrences(env) >= 3;
}

// ============================================================================
// Attack detection (bitboard version)
// ============================================================================

// Check if square sq is attacked by any piece belonging to by_player
static inline int is_square_attacked(ChessEnv* env, int sq, int by_player) {
    Bitboard occ = env->bb_occ;
    Bitboard attackers = env->bb_by_color[by_player];

    // Pawn attacks: look from the target square using opposite color's attack pattern
    if (PawnAttacksBB[1 - by_player][sq] & attackers & env->bb_by_type[BB_PAWN]) return 1;
    if (KnightAttacksBB[sq] & attackers & env->bb_by_type[BB_KNIGHT]) return 1;
    if (KingAttacksBB[sq] & attackers & env->bb_by_type[BB_KING]) return 1;
    Bitboard bishop_queen = env->bb_by_type[BB_BISHOP] | env->bb_by_type[BB_QUEEN];
    if (bishop_attacks_bb(sq, occ) & attackers & bishop_queen) return 1;
    Bitboard rook_queen = env->bb_by_type[BB_ROOK] | env->bb_by_type[BB_QUEEN];
    if (rook_attacks_bb(sq, occ) & attackers & rook_queen) return 1;
    return 0;
}

// Find king square for given player
static inline int find_king(ChessEnv* env, int player) {
    Bitboard king_bb = env->bb_by_color[player] & env->bb_by_type[BB_KING];
    if (king_bb == 0) return -1;
    return bb_lsb(king_bb);
}

// Check if player's king is in check
static inline int is_in_check(ChessEnv* env, int player) {
    int king_sq = find_king(env, player);
    if (king_sq < 0) return 0;
    int opponent = 1 - player;
    return is_square_attacked(env, king_sq, opponent);
}

// ============================================================================
// Move generation (bitboard version)
// ============================================================================

// Try adding a move; returns 1 if added, 0 if buffer full
static inline int add_move(int moves[], int* count, int max_moves, int from_sq, int to_sq) {
    if (*count >= max_moves) return 0;
    moves[(*count)++] = from_sq * 64 + to_sq;
    return 1;
}

// Generate all pseudo-legal moves for current player using bitboards
static int generate_pseudo_legal_moves(ChessEnv* env, int moves[], int max_moves) {
    int count = 0;
    int player = env->current_player;
    Bitboard own = env->bb_by_color[player];
    Bitboard enemy = env->bb_by_color[1 - player];
    Bitboard occ = env->bb_occ;
    Bitboard empty_sq = ~occ;
    Bitboard target = ~own;  // can go to empty or enemy squares

    // === PAWNS ===
    Bitboard pawns = own & env->bb_by_type[BB_PAWN];
    if (player == 0) {
        // White: push north (shift left by 8)
        Bitboard single = (pawns << 8) & empty_sq;
        Bitboard dbl = ((single & Rank3BB) << 8) & empty_sq;
        while (single) {
            int to = bb_pop_lsb(&single);
            add_move(moves, &count, max_moves, to - 8, to);
        }
        while (dbl) {
            int to = bb_pop_lsb(&dbl);
            add_move(moves, &count, max_moves, to - 16, to);
        }
        // Captures
        Bitboard cap_left = (pawns << 7) & enemy & ~FileHBB;
        Bitboard cap_right = (pawns << 9) & enemy & ~FileABB;
        while (cap_left) {
            int to = bb_pop_lsb(&cap_left);
            add_move(moves, &count, max_moves, to - 7, to);
        }
        while (cap_right) {
            int to = bb_pop_lsb(&cap_right);
            add_move(moves, &count, max_moves, to - 9, to);
        }
        // En passant
        if (env->en_passant_square >= 0) {
            Bitboard ep_bb = 1ULL << env->en_passant_square;
            Bitboard ep_left = (pawns << 7) & ep_bb & ~FileHBB;
            Bitboard ep_right = (pawns << 9) & ep_bb & ~FileABB;
            if (ep_left) {
                add_move(moves, &count, max_moves, env->en_passant_square - 7, env->en_passant_square);
            }
            if (ep_right) {
                add_move(moves, &count, max_moves, env->en_passant_square - 9, env->en_passant_square);
            }
        }
    } else {
        // Black: push south (shift right by 8)
        Bitboard single = (pawns >> 8) & empty_sq;
        Bitboard dbl = ((single & Rank6BB) >> 8) & empty_sq;
        while (single) {
            int to = bb_pop_lsb(&single);
            add_move(moves, &count, max_moves, to + 8, to);
        }
        while (dbl) {
            int to = bb_pop_lsb(&dbl);
            add_move(moves, &count, max_moves, to + 16, to);
        }
        // Captures
        Bitboard cap_left = (pawns >> 9) & enemy & ~FileHBB;
        Bitboard cap_right = (pawns >> 7) & enemy & ~FileABB;
        while (cap_left) {
            int to = bb_pop_lsb(&cap_left);
            add_move(moves, &count, max_moves, to + 9, to);
        }
        while (cap_right) {
            int to = bb_pop_lsb(&cap_right);
            add_move(moves, &count, max_moves, to + 7, to);
        }
        // En passant
        if (env->en_passant_square >= 0) {
            Bitboard ep_bb = 1ULL << env->en_passant_square;
            Bitboard ep_left = (pawns >> 9) & ep_bb & ~FileHBB;
            Bitboard ep_right = (pawns >> 7) & ep_bb & ~FileABB;
            if (ep_left) {
                add_move(moves, &count, max_moves, env->en_passant_square + 9, env->en_passant_square);
            }
            if (ep_right) {
                add_move(moves, &count, max_moves, env->en_passant_square + 7, env->en_passant_square);
            }
        }
    }

    // === KNIGHTS ===
    Bitboard knights = own & env->bb_by_type[BB_KNIGHT];
    while (knights) {
        int from = bb_pop_lsb(&knights);
        Bitboard atk = KnightAttacksBB[from] & target;
        while (atk) {
            add_move(moves, &count, max_moves, from, bb_pop_lsb(&atk));
        }
    }

    // === BISHOPS ===
    Bitboard bishops = own & env->bb_by_type[BB_BISHOP];
    while (bishops) {
        int from = bb_pop_lsb(&bishops);
        Bitboard atk = bishop_attacks_bb(from, occ) & target;
        while (atk) {
            add_move(moves, &count, max_moves, from, bb_pop_lsb(&atk));
        }
    }

    // === ROOKS ===
    Bitboard rooks = own & env->bb_by_type[BB_ROOK];
    while (rooks) {
        int from = bb_pop_lsb(&rooks);
        Bitboard atk = rook_attacks_bb(from, occ) & target;
        while (atk) {
            add_move(moves, &count, max_moves, from, bb_pop_lsb(&atk));
        }
    }

    // === QUEENS ===
    Bitboard queens = own & env->bb_by_type[BB_QUEEN];
    while (queens) {
        int from = bb_pop_lsb(&queens);
        Bitboard atk = queen_attacks_bb(from, occ) & target;
        while (atk) {
            add_move(moves, &count, max_moves, from, bb_pop_lsb(&atk));
        }
    }

    // === KING ===
    Bitboard king_bb = own & env->bb_by_type[BB_KING];
    if (king_bb) {
        int king_sq = bb_lsb(king_bb);
        Bitboard king_atk = KingAttacksBB[king_sq] & target;
        while (king_atk) {
            add_move(moves, &count, max_moves, king_sq, bb_pop_lsb(&king_atk));
        }

        // === CASTLING ===
        int opponent = 1 - player;
        if (player == 0) {
            if ((env->castling_rights & CASTLE_WK) &&
                env->board[5] == EMPTY && env->board[6] == EMPTY &&
                !is_square_attacked(env, 4, opponent) &&
                !is_square_attacked(env, 5, opponent) &&
                !is_square_attacked(env, 6, opponent)) {
                add_move(moves, &count, max_moves, 4, 6);
            }
            if ((env->castling_rights & CASTLE_WQ) &&
                env->board[3] == EMPTY && env->board[2] == EMPTY && env->board[1] == EMPTY &&
                !is_square_attacked(env, 4, opponent) &&
                !is_square_attacked(env, 3, opponent) &&
                !is_square_attacked(env, 2, opponent)) {
                add_move(moves, &count, max_moves, 4, 2);
            }
        } else {
            if ((env->castling_rights & CASTLE_BK) &&
                env->board[61] == EMPTY && env->board[62] == EMPTY &&
                !is_square_attacked(env, 60, opponent) &&
                !is_square_attacked(env, 61, opponent) &&
                !is_square_attacked(env, 62, opponent)) {
                add_move(moves, &count, max_moves, 60, 62);
            }
            if ((env->castling_rights & CASTLE_BQ) &&
                env->board[59] == EMPTY && env->board[58] == EMPTY && env->board[57] == EMPTY &&
                !is_square_attacked(env, 60, opponent) &&
                !is_square_attacked(env, 59, opponent) &&
                !is_square_attacked(env, 58, opponent)) {
                add_move(moves, &count, max_moves, 60, 58);
            }
        }
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

    // Save bitboard state for temporary move
    Bitboard save_bb_type[7], save_bb_color[2], save_bb_occ;
    memcpy(save_bb_type, env->bb_by_type, sizeof(save_bb_type));
    memcpy(save_bb_color, env->bb_by_color, sizeof(save_bb_color));
    save_bb_occ = env->bb_occ;
    uint64_t save_zobrist = env->zobrist_key;
    int save_piece_count[2][6];
    memcpy(save_piece_count, env->piece_count, sizeof(save_piece_count));

    // Apply move temporarily to bitboards too
    if (ep_capture_sq >= 0) {
        bb_remove_piece(env, ep_capture_sq, ep_captured);
    }
    if (captured != EMPTY) {
        bb_remove_piece(env, to_sq, captured);
    }
    bb_move_piece(env, from_sq, to_sq, moved);

    // Make the move temporarily on board
    env->board[to_sq] = moved;
    env->board[from_sq] = EMPTY;

    // Handle castling king move - need to also move the rook temporarily
    int rook_from = -1, rook_to = -1;
    int8_t rook_piece = EMPTY;
    if (ptype == WK) {
        if (from_sq == 4 && to_sq == 6) {
            rook_from = 7; rook_to = 5; rook_piece = WR;
        } else if (from_sq == 4 && to_sq == 2) {
            rook_from = 0; rook_to = 3; rook_piece = WR;
        } else if (from_sq == 60 && to_sq == 62) {
            rook_from = 63; rook_to = 61; rook_piece = BR;
        } else if (from_sq == 60 && to_sq == 58) {
            rook_from = 56; rook_to = 59; rook_piece = BR;
        }
        if (rook_from >= 0) {
            env->board[rook_to] = rook_piece;
            env->board[rook_from] = EMPTY;
            bb_move_piece(env, rook_from, rook_to, rook_piece);
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

    // Restore bitboard state
    memcpy(env->bb_by_type, save_bb_type, sizeof(save_bb_type));
    memcpy(env->bb_by_color, save_bb_color, sizeof(save_bb_color));
    env->bb_occ = save_bb_occ;
    env->zobrist_key = save_zobrist;
    memcpy(env->piece_count, save_piece_count, sizeof(save_piece_count));

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

// Generate legal moves with caching using Zobrist key
static int generate_legal_moves_cached(ChessEnv* env, int moves[], int max_moves) {
    uint64_t key = env->zobrist_key;
    int side = env->current_player;

    // Cache hit: same position hash and same side to move
    if (env->legal_moves_cache_count >= 0 &&
        env->legal_moves_key == key &&
        env->legal_moves_side == side) {
        int count = env->legal_moves_cache_count;
        if (count > max_moves) count = max_moves;
        memcpy(moves, env->legal_moves_cache, count * sizeof(int));
        return count;
    }

    // Cache miss: generate and store
    int count = generate_legal_moves(env, env->legal_moves_cache, CHESS_MAX_MOVES);
    env->legal_moves_cache_count = count;
    env->legal_moves_key = key;
    env->legal_moves_side = side;

    int ret = count;
    if (ret > max_moves) ret = max_moves;
    memcpy(moves, env->legal_moves_cache, ret * sizeof(int));
    return ret;
}

// Early-exit: returns 1 as soon as any legal move is found, 0 if none exist.
static int has_any_legal_move(ChessEnv* env) {
    // Check cache first
    uint64_t key = env->zobrist_key;
    int side = env->current_player;
    if (key == env->legal_moves_key && side == env->legal_moves_side &&
        env->legal_moves_cache_count >= 0) {
        return env->legal_moves_cache_count > 0;
    }

    // Cache miss: do the full pseudo-legal scan
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
// Move application (with bitboard + Zobrist maintenance)
// ============================================================================

// Apply a move with explicit promotion piece type.
static void apply_move_ex(ChessEnv* env, int from_sq, int to_sq, int8_t promo_piece) {
    int8_t piece = env->board[from_sq];
    int8_t captured = env->board[to_sq];
    int player = env->current_player;

    int ptype = piece;
    if (ptype >= BP) ptype -= 6;

    int is_capture = (captured != EMPTY);

    // Zobrist: XOR out old castling/EP state
    env->zobrist_key ^= zob.castling[env->castling_rights];
    if (env->en_passant_square >= 0)
        env->zobrist_key ^= zob.enpassant[sq_col(env->en_passant_square)];

    // Handle en passant capture
    if (ptype == WP && to_sq == env->en_passant_square) {
        int dir = (player == 0) ? -1 : 1;
        int ep_cap_sq = to_sq + dir * 8;
        int8_t ep_captured = env->board[ep_cap_sq];
        env->board[ep_cap_sq] = EMPTY;
        bb_remove_piece(env, ep_cap_sq, ep_captured);
        is_capture = 1;
    }

    // Capture on destination
    if (captured != EMPTY) {
        bb_remove_piece(env, to_sq, captured);
    }

    // Move piece on bitboards
    bb_move_piece(env, from_sq, to_sq, piece);

    // Move piece on board
    env->board[to_sq] = piece;
    env->board[from_sq] = EMPTY;

    // Pawn promotion
    if (ptype == WP) {
        int promo_row = (player == 0) ? 7 : 0;
        if (sq_row(to_sq) == promo_row) {
            int8_t promo;
            if (promo_piece != EMPTY) {
                promo = promo_piece;
            } else {
                promo = (player == 0) ? WQ : BQ;
            }
            // Remove pawn from bitboards, add promoted piece
            bb_remove_piece(env, to_sq, piece);
            bb_add_piece(env, to_sq, promo);
            env->board[to_sq] = promo;
        }
    }

    // Handle castling rook movement
    if (ptype == WK) {
        if (from_sq == 4 && to_sq == 6) {       // White kingside
            bb_move_piece(env, 7, 5, WR);
            env->board[5] = WR;
            env->board[7] = EMPTY;
        } else if (from_sq == 4 && to_sq == 2) { // White queenside
            bb_move_piece(env, 0, 3, WR);
            env->board[3] = WR;
            env->board[0] = EMPTY;
        } else if (from_sq == 60 && to_sq == 62) { // Black kingside
            bb_move_piece(env, 63, 61, BR);
            env->board[61] = BR;
            env->board[63] = EMPTY;
        } else if (from_sq == 60 && to_sq == 58) { // Black queenside
            bb_move_piece(env, 56, 59, BR);
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
            env->en_passant_square = (from_sq + to_sq) / 2;
        }
    }

    // Zobrist: XOR in new castling/EP state + flip side
    env->zobrist_key ^= zob.castling[env->castling_rights];
    if (env->en_passant_square >= 0)
        env->zobrist_key ^= zob.enpassant[sq_col(env->en_passant_square)];
    env->zobrist_key ^= zob.side;

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
// Game end detection (uses piece_count for insufficient material)
// ============================================================================

static int check_game_end(ChessEnv* env, int has_legal) {
    // Threefold repetition
    if (env->enable_threefold_repetition && check_threefold_repetition(env)) {
        return GAME_REPETITION;
    }

    // Fifty-move rule
    if (env->enable_50_move_rule && env->halfmove_clock >= 100) {
        return GAME_FIFTY_MOVE;
    }

    // has_legal: 1 if current player has at least one legal move, 0 if none
    if (!has_legal) {
        if (is_in_check(env, env->current_player)) {
            return GAME_CHECKMATE;
        }
        return GAME_STALEMATE;
    }

    // Insufficient material using piece_count
    // piece_count[color][0]=Pawn, [1]=Knight, [2]=Bishop, [3]=Rook, [4]=Queen, [5]=King
    int wp = env->piece_count[0][0]; // white pawns
    int wn = env->piece_count[0][1]; int wb = env->piece_count[0][2];
    int wr = env->piece_count[0][3]; int wq = env->piece_count[0][4];
    int bp = env->piece_count[1][0];
    int bn = env->piece_count[1][1]; int bb_cnt = env->piece_count[1][2];
    int br = env->piece_count[1][3]; int bq = env->piece_count[1][4];

    int white_others = wp + wr + wq;
    int black_others = bp + br + bq;

    if (white_others == 0 && black_others == 0) {
        int wminor = wn + wb;
        int bminor = bn + bb_cnt;
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
// 1-agent topology: all rewards go to env->rewards[0] with sign based on learner_color.
static int process_player_action(ChessEnv* env, int action, int player) {
    PhaseState* ps = &env->phase_state[0]; // single phase state slot
    float sign = (player == env->learner_color) ? 1.0f : -1.0f;

    // PASS action — in 1-agent mode, it's always the mover's turn, so PASS is invalid
    if (action == CHESS_PASS_ACTION) {
        env->rewards[0] += sign * env->reward_invalid_move;
        env->episode_illegal_moves += 1.0f;
        return 0;
    }

    if (ps->pick_phase == 0) {
        // Phase 0: Pick a piece
        if (action < 0 || action > 63) {
            env->rewards[0] += sign * env->reward_invalid_piece;
            env->episode_illegal_moves += 1.0f;
            return 0;
        }

        // Obs is always from current mover's perspective (possibly flipped)
        int abs_sq = (player == 0) ? action : flip_sq(action);

        if (!is_own_piece(env->board[abs_sq], player)) {
            env->rewards[0] += sign * env->reward_invalid_piece;
            env->episode_illegal_moves += 1.0f;
            return 0;
        }

        int all_legal[CHESS_MAX_MOVES];
        int num_legal = generate_legal_moves_cached(env, all_legal, CHESS_MAX_MOVES);

        ps->valid_dest_count = 0;
        for (int i = 0; i < num_legal; i++) {
            int from = all_legal[i] / 64;
            if (from == abs_sq) {
                ps->valid_dest_moves[ps->valid_dest_count++] = all_legal[i];
            }
        }

        if (ps->valid_dest_count == 0) {
            env->rewards[0] += sign * env->reward_invalid_piece;
            env->episode_illegal_moves += 1.0f;
            return 0;
        }

        ps->selected_square = abs_sq;
        ps->pick_phase = 1;
        env->rewards[0] += sign * env->reward_valid_piece;
        return 0;

    } else {
        // Phase 1: Pick destination or promotion
        int from_sq = ps->selected_square;
        int to_sq = -1;
        int8_t promo_piece_val = EMPTY;

        if (action >= 0 && action <= 63) {
            to_sq = (player == 0) ? action : flip_sq(action);
        } else if (action >= 64 && action <= 95) {
            int promo_idx = action - 64;
            int promo_type = promo_idx / 8;
            int promo_file = promo_idx % 8;

            int promo_row = (player == 0) ? 7 : 0;
            to_sq = make_sq(promo_row, promo_file);

            int8_t promo_types_white[4] = {WQ, WR, WB, WN};
            int8_t promo_types_black[4] = {BQ, BR, BB_PIECE, BN};
            promo_piece_val = (player == 0) ? promo_types_white[promo_type] : promo_types_black[promo_type];
        } else {
            ps->pick_phase = 0;
            ps->selected_square = -1;
            env->rewards[0] += sign * env->reward_invalid_move;
            env->episode_illegal_moves += 1.0f;
            return 0;
        }

        int move_encoded = from_sq * 64 + to_sq;
        int found = 0;
        for (int i = 0; i < ps->valid_dest_count; i++) {
            if (ps->valid_dest_moves[i] == move_encoded) {
                found = 1;
                break;
            }
        }

        if (!found) {
            ps->pick_phase = 0;
            ps->selected_square = -1;
            env->rewards[0] += sign * env->reward_invalid_move;
            env->episode_illegal_moves += 1.0f;
            return 0;
        }

        // Capture bonus: check what's on destination before applying
        int8_t cap_piece = env->board[to_sq];
        int8_t moving = env->board[from_sq];
        int mtype = moving;
        if (mtype >= BP) mtype -= 6;
        if (mtype == WP && to_sq == env->en_passant_square) {
            cap_piece = (player == 0) ? BP : WP;
        }

        int opp_player = 1 - player;
        int mat_before = compute_material_score(env, player) - compute_material_score(env, opp_player);
        int pos_before = compute_positional_score(env, player) - compute_positional_score(env, opp_player);

        int is_castling = 0;
        if (mtype == WK) {
            int from_col_val = sq_col(from_sq);
            int to_col_val = sq_col(to_sq);
            int col_diff = to_col_val - from_col_val;
            if (col_diff < 0) col_diff = -col_diff;
            if (col_diff == 2) is_castling = 1;
        }

        int see_value = 0;
        int is_capture = (cap_piece != EMPTY);
        if (env->reward_see_hanging != 0.0f) {
            if (is_capture) {
                see_value = see_capture(env, from_sq, to_sq, moving, cap_piece, player);
            } else {
                see_value = see_square(env, from_sq, to_sq, moving, player);
            }
            env->last_see_value = see_value;
        }

        // Apply move
        apply_move_ex(env, from_sq, to_sq, promo_piece_val);

        ps->pick_phase = 0;
        ps->selected_square = -1;
        env->rewards[0] += sign * env->reward_valid_move;

        if (cap_piece != EMPTY && env->reward_capture_bonus != 0.0f) {
            int cap_val = captured_piece_value(cap_piece);
            env->rewards[0] += sign * env->reward_capture_bonus * (float)cap_val / 900.0f;
        }

        if (env->reward_check_bonus != 0.0f) {
            int opp = 1 - player;
            if (is_in_check(env, opp)) {
                env->rewards[0] += sign * env->reward_check_bonus;
            }
        }

        if (env->reward_material != 0.0f) {
            int mat_after = compute_material_score(env, player) - compute_material_score(env, opp_player);
            int mat_delta = mat_after - mat_before;
            env->rewards[0] += sign * env->reward_material * (float)mat_delta / 100.0f;
        }

        if (env->reward_position != 0.0f) {
            int pos_after = compute_positional_score(env, player) - compute_positional_score(env, opp_player);
            int pos_delta = pos_after - pos_before;
            env->rewards[0] += sign * env->reward_position * (float)pos_delta / 100.0f;
        }

        if (is_castling && env->reward_castling != 0.0f) {
            env->rewards[0] += sign * env->reward_castling;
        }

        if (env->reward_see_hanging != 0.0f) {
            if (is_capture && see_value < 0) {
                if (env->reward_material != 0.0f) {
                    int mat_after_2 = compute_material_score(env, player) - compute_material_score(env, opp_player);
                    int mat_delta_2 = mat_after_2 - mat_before;
                    if (mat_delta_2 > 0) {
                        env->rewards[0] -= sign * env->reward_material * (float)mat_delta_2 / 100.0f;
                    }
                }
                env->rewards[0] += sign * env->reward_see_hanging * (float)(-see_value) / 100.0f;
            } else if (!is_capture && see_value < 0) {
                env->rewards[0] += sign * env->reward_see_hanging * (float)(-see_value) / 100.0f;
            }
        }

        return 1;
    }
}

// ============================================================================
// Observation writing
// ============================================================================

// 1-agent topology: valid pieces mask for current mover
static void compute_valid_pieces_mask(ChessEnv* env, unsigned char* mask) {
    memset(mask, 0, 64);

    int player = env->current_player;
    int all_legal[CHESS_MAX_MOVES];
    int num_legal = generate_legal_moves_cached(env, all_legal, CHESS_MAX_MOVES);

    unsigned char has_legal_from[64];
    memset(has_legal_from, 0, 64);
    for (int i = 0; i < num_legal; i++) {
        int from = all_legal[i] / 64;
        has_legal_from[from] = 1;
    }

    for (int sq = 0; sq < 64; sq++) {
        if (has_legal_from[sq]) {
            int psq = (player == 0) ? sq : flip_sq(sq);
            mask[psq] = 255;
        }
    }
}

// 1-agent topology: valid dests for current mover
static void compute_valid_dests_mask(ChessEnv* env, unsigned char* dest_mask, unsigned char* promo_mask) {
    memset(dest_mask, 0, 64);
    memset(promo_mask, 0, 32);

    int player = env->current_player;
    PhaseState* ps = &env->phase_state[0]; // single phase state slot
    if (ps->pick_phase != 1) return;

    int from_sq = ps->selected_square;

    for (int i = 0; i < ps->valid_dest_count; i++) {
        int to = ps->valid_dest_moves[i] % 64;
        int psq = (player == 0) ? to : flip_sq(to);

        if (is_promotion_move(env, from_sq, to)) {
            int file = sq_col(to);
            for (int pt = 0; pt < 4; pt++) {
                promo_mask[pt * 8 + file] = 255;
            }
            dest_mask[psq] = 255;
        } else {
            dest_mask[psq] = 255;
        }
    }
}

// 1-agent topology: write single observation from current mover's perspective.
static void write_observations(ChessEnv* env) {
    unsigned char* obs = env->observations; // single obs buffer
    int viewer = env->current_player;

    // === Board (offset 0, size 64) ===
    if (viewer == 0) {
        // White: normal orientation
        for (int sq = 0; sq < 64; sq++) {
            obs[sq] = (unsigned char)env->board[sq];
        }
    } else {
        // Black: flip rows + swap colors
        for (int sq = 0; sq < 64; sq++) {
            int flipped_sq = flip_sq(sq);
            int8_t p = env->board[flipped_sq];
            unsigned char obs_val;
            if (p == EMPTY) {
                obs_val = 0;
            } else if (is_white_piece(p)) {
                obs_val = (unsigned char)(p + 6); // White pieces appear as Black
            } else {
                obs_val = (unsigned char)(p - 6); // Black pieces appear as White
            }
            obs[sq] = obs_val;
        }
    }

    // === Side to move one-hot (offset 64, size 2) ===
    // Always [255, 0] since obs is always from current mover's perspective
    obs[64] = 255;
    obs[65] = 0;

    // === Castling rights (offset 66, size 4) — from viewer's perspective ===
    if (viewer == 0) {
        obs[66] = (env->castling_rights & CASTLE_WK) ? 255 : 0;
        obs[67] = (env->castling_rights & CASTLE_WQ) ? 255 : 0;
        obs[68] = (env->castling_rights & CASTLE_BK) ? 255 : 0;
        obs[69] = (env->castling_rights & CASTLE_BQ) ? 255 : 0;
    } else {
        obs[66] = (env->castling_rights & CASTLE_BK) ? 255 : 0;
        obs[67] = (env->castling_rights & CASTLE_BQ) ? 255 : 0;
        obs[68] = (env->castling_rights & CASTLE_WK) ? 255 : 0;
        obs[69] = (env->castling_rights & CASTLE_WQ) ? 255 : 0;
    }

    // === En passant file (offset 70, size 1) ===
    if (env->en_passant_square >= 0) {
        obs[70] = (unsigned char)sq_col(env->en_passant_square);
    } else {
        obs[70] = 255;
    }

    // === Phase one-hot (offset 71, size 2) ===
    PhaseState* ps = &env->phase_state[0]; // single phase state slot
    obs[71] = (ps->pick_phase == 0) ? 255 : 0;
    obs[72] = (ps->pick_phase == 1) ? 255 : 0;

    // === Selected piece plane (offset 73, size 64) ===
    memset(obs + 73, 0, 64);
    if (ps->pick_phase == 1 && ps->selected_square >= 0) {
        int psq = (viewer == 0) ? ps->selected_square : flip_sq(ps->selected_square);
        obs[73 + psq] = 255;
    }

    // === Valid pieces mask (offset 137, size 64) ===
    compute_valid_pieces_mask(env, obs + 137);

    // === Valid destinations mask (offset 201, size 64) + Valid promotions (offset 265, size 32) ===
    compute_valid_dests_mask(env, obs + 201, obs + 265);

    // === Self in check (offset 297, size 1) ===
    obs[297] = is_in_check(env, viewer) ? 255 : 0;

    // === Opponent in check (offset 298, size 1) ===
    obs[298] = is_in_check(env, 1 - viewer) ? 255 : 0;

    // === Rule50 counter (offset 299, size 1) ===
    int rule50_scaled = (env->halfmove_clock * 255) / 100;
    if (rule50_scaled > 255) rule50_scaled = 255;
    obs[299] = (unsigned char)rule50_scaled;

    // === Pass valid (offset 300, size 1) ===
    // In 1-agent mode, it's always the mover's turn, so PASS is never valid
    obs[300] = 0;
}

// ============================================================================
// Board setup
// ============================================================================

static void setup_initial_board(ChessEnv* env) {
    memset(env->board, EMPTY, 64);

    env->board[0] = WR;
    env->board[1] = WN;
    env->board[2] = WB;
    env->board[3] = WQ;
    env->board[4] = WK;
    env->board[5] = WB;
    env->board[6] = WN;
    env->board[7] = WR;

    for (int c = 0; c < 8; c++) {
        env->board[make_sq(1, c)] = WP;
    }

    for (int c = 0; c < 8; c++) {
        env->board[make_sq(6, c)] = BP;
    }

    env->board[56] = BR;
    env->board[57] = BN;
    env->board[58] = BB_PIECE;
    env->board[59] = BQ;
    env->board[60] = BK;
    env->board[61] = BB_PIECE;
    env->board[62] = BN;
    env->board[63] = BR;
}

// ============================================================================
// FEN parser
// ============================================================================

static int setup_from_fen(ChessEnv* env, const char* fen) {
    if (!fen || !*fen) return 0;

    memset(env->board, EMPTY, 64);

    int row = 7;
    int col = 0;
    const char* p = fen;

    while (*p && *p != ' ') {
        if (*p == '/') {
            row--;
            col = 0;
            if (row < 0) return 0;
        } else if (*p >= '1' && *p <= '8') {
            col += (*p - '0');
        } else {
            if (row < 0 || row > 7 || col < 0 || col > 7) return 0;
            int sq = make_sq(row, col);
            switch (*p) {
                case 'P': env->board[sq] = WP; break;
                case 'N': env->board[sq] = WN; break;
                case 'B': env->board[sq] = WB; break;
                case 'R': env->board[sq] = WR; break;
                case 'Q': env->board[sq] = WQ; break;
                case 'K': env->board[sq] = WK; break;
                case 'p': env->board[sq] = BP; break;
                case 'n': env->board[sq] = BN; break;
                case 'b': env->board[sq] = BB_PIECE; break;
                case 'r': env->board[sq] = BR; break;
                case 'q': env->board[sq] = BQ; break;
                case 'k': env->board[sq] = BK; break;
                default: return 0;
            }
            col++;
        }
        p++;
    }

    if (*p != ' ') return 0;
    p++;

    if (*p == 'w') {
        env->current_player = 0;
    } else if (*p == 'b') {
        env->current_player = 1;
    } else {
        return 0;
    }
    p++;

    if (*p != ' ') return 0;
    p++;

    env->castling_rights = 0;
    if (*p == '-') {
        p++;
    } else {
        while (*p && *p != ' ') {
            switch (*p) {
                case 'K': env->castling_rights |= CASTLE_WK; break;
                case 'Q': env->castling_rights |= CASTLE_WQ; break;
                case 'k': env->castling_rights |= CASTLE_BK; break;
                case 'q': env->castling_rights |= CASTLE_BQ; break;
                default: break;
            }
            p++;
        }
    }

    if (*p != ' ') return 0;
    p++;

    env->en_passant_square = -1;
    if (*p == '-') {
        p++;
    } else {
        if (*p >= 'a' && *p <= 'h' && *(p + 1) >= '1' && *(p + 1) <= '8') {
            int file = *p - 'a';
            int rank = *(p + 1) - '1';
            env->en_passant_square = make_sq(rank, file);
            p += 2;
        } else {
            return 0;
        }
    }

    env->halfmove_clock = 0;
    if (*p == ' ') {
        p++;
        env->halfmove_clock = 0;
        while (*p >= '0' && *p <= '9') {
            env->halfmove_clock = env->halfmove_clock * 10 + (*p - '0');
            p++;
        }
    }

    env->fullmove_number = 1;
    if (*p == ' ') {
        p++;
        env->fullmove_number = 0;
        while (*p >= '0' && *p <= '9') {
            env->fullmove_number = env->fullmove_number * 10 + (*p - '0');
            p++;
        }
        if (env->fullmove_number == 0) env->fullmove_number = 1;
    }

    // Sync bitboards after FEN setup
    sync_bitboards_from_board(env);

    return 1;
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
    env->reward_capture_bonus = 0.0f;
    env->reward_check_bonus = 0.0f;
    env->reward_repetition = 0.0f;
    env->reward_material = 0.0f;
    env->reward_position = 0.0f;
    env->reward_castling = 0.0f;
    env->reward_draw = 0.0f;
    env->reward_see_hanging = 0.0f;
    env->enable_50_move_rule = 1;
    env->enable_threefold_repetition = 1;
    env->use_curriculum = 0;
}

void c_reset(ChessEnv* env) {
    setup_initial_board(env);
    env->current_player = 0;
    env->castling_rights = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    env->en_passant_square = -1;
    env->halfmove_clock = 0;
    env->fullmove_number = 1;
    env->step_count = 0;

    // Sync bitboards from board
    sync_bitboards_from_board(env);

    // Alternate learner_color each reset
    env->learner_color = 1 - env->learner_color;

    // Reset single phase state slot (1-agent topology)
    env->phase_state[0].pick_phase = 0;
    env->phase_state[0].selected_square = -1;
    env->phase_state[0].valid_dest_count = 0;

    // Clear single reward and terminal slot
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    // Reset per-episode counters
    env->episode_illegal_moves = 0.0f;
    env->episode_chess_moves = 0;

    // Invalidate legal move cache
    env->legal_moves_cache_count = -1;
    env->legal_moves_key = 0;
    env->legal_moves_side = -1;

    // Reset position history and record initial position
    env->position_history_count = 0;
    record_position_hash(env);

    write_observations(env);
}

static void log_episode(ChessEnv* env) {
    env->log.episode_length += (float)env->step_count;
    env->log.episode_return += env->rewards[0];
    env->log.illegal_moves += env->episode_illegal_moves;
    // Material/positional from learner's perspective
    int lc = env->learner_color;
    env->log.material_score += (float)(compute_material_score(env, lc) - compute_material_score(env, 1 - lc));
    env->log.positional_score += (float)(compute_positional_score(env, lc) - compute_positional_score(env, 1 - lc));
    env->log.invalid_action_rate += (env->step_count > 0) ? env->episode_illegal_moves / (float)env->step_count : 0.0f;
    env->log.chess_moves += (float)env->episode_chess_moves;
    env->log.n += 1;
}

// 1-agent topology: single action per step, always from current mover.
void c_step(ChessEnv* env) {
    // Auto-reset: if terminal at start of step, reset and return
    if (env->terminals[0] == 1) {
        c_reset(env);
        return;
    }

    // Clear single reward
    env->rewards[0] = 0.0f;

    // Read single action from PPO
    int action = get_action(env, 0);
    if (action < 0 || action >= CHESS_NUM_ACTIONS) action = 0;

    int player = env->current_player;

    int move_made = process_player_action(env, action, player);

    if (move_made) {
        env->episode_chess_moves++;
        env->current_player = 1 - env->current_player;
        record_position_hash(env);

        // Repetition penalty
        if (env->reward_repetition != 0.0f) {
            int occ = count_position_occurrences(env);
            if (occ >= 2) {
                float rep_sign = (player == env->learner_color) ? 1.0f : -1.0f;
                env->rewards[0] += rep_sign * env->reward_repetition;
            }
        }

        int result = check_game_end(env, has_any_legal_move(env));

        if (result == GAME_CHECKMATE) {
            // player (the mover who just moved) wins
            if (player == env->learner_color) {
                env->rewards[0] += 1.0f;
            } else {
                env->rewards[0] += -1.0f;
            }
            env->terminals[0] = 1;

            if (player == 0) {
                env->log.white_wins += 1;
            } else {
                env->log.black_wins += 1;
            }
            log_episode(env);
        } else if (result == GAME_REPETITION) {
            env->rewards[0] += env->reward_draw;
            env->terminals[0] = 1;
            env->log.draws += 1;
            env->log.repetitions += 1;
            log_episode(env);
        } else if (result == GAME_STALEMATE || result == GAME_FIFTY_MOVE || result == GAME_INSUFFICIENT) {
            env->rewards[0] += env->reward_draw;
            env->terminals[0] = 1;
            env->log.draws += 1;
            log_episode(env);
        }
    }

    write_observations(env);

    env->step_count++;
    if (env->step_count >= env->max_steps && env->terminals[0] == 0) {
        env->rewards[0] += env->reward_draw;
        env->terminals[0] = 1;
        env->log.draws += 1;
        log_episode(env);
    }
}

void c_close(ChessEnv* env) {
    (void)env;
}

#endif // CHESS_H
