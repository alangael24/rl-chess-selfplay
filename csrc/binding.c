/*
 * binding.c - Custom PufferLib C binding for Chess self-play.
 *
 * 1-agent-per-game topology: each game has exactly 1 agent slot.
 * The agent controls whoever's turn it is (White or Black).
 *
 * Layout in PufferLib buffers (for N games, N agent slots):
 *   observations[i]  = obs for game i (from current mover's perspective)
 *   actions[i]       = action for game i
 *   rewards[i]       = reward for game i (signed: + for learner, - for opponent)
 *   terminals[i]     = terminal for game i
 *
 * ChessEnv.observations -> &observations[i * OBS_SIZE]
 * ChessEnv.actions      -> &actions[i]
 * ChessEnv.rewards      -> &rewards[i]
 * ChessEnv.terminals    -> &terminals[i]
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "chess.h"

typedef struct {
    ChessEnv* games;
    int num_games;
    int num_agents;  /* = num_games (1 agent per game) */
    char** fen_list;
    int fen_count;
    float fen_curric_pct;
} VecEnv;

#define QPOL_FILE_MAGIC "NNUV1\0\0"
#define QPOL_FILE_MAGIC_SIZE 8

static int read_exact(FILE* f, void* dst, size_t nbytes) {
    return fread(dst, 1, nbytes, f) == nbytes;
}

static int load_qpolicy_file(const char* path, char* err, size_t err_sz) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(err, err_sz, "Failed to open qpolicy file: %s", path);
        return 0;
    }

    char magic[QPOL_FILE_MAGIC_SIZE];
    int32_t dims[3];
    int32_t search_depth = CHESS_QPOL_SEARCH_DEPTH_DEFAULT;
    int32_t out_b = 0;

    qpol_clear();
    if (!read_exact(f, magic, sizeof(magic))) {
        snprintf(err, err_sz, "qpolicy file too short (magic)");
        fclose(f);
        return 0;
    }
    if (memcmp(magic, QPOL_FILE_MAGIC, QPOL_FILE_MAGIC_SIZE) != 0) {
        snprintf(err, err_sz, "Invalid qpolicy magic");
        fclose(f);
        return 0;
    }
    if (!read_exact(f, dims, sizeof(dims))) {
        snprintf(err, err_sz, "qpolicy file too short (dims)");
        fclose(f);
        return 0;
    }
    if (dims[0] != CHESS_HALFKP_FEATURES || dims[1] != CHESS_NNUE_ACCUM || dims[2] != CHESS_NNUE_HIDDEN) {
        snprintf(err, err_sz, "qpolicy dims mismatch: got (%d,%d,%d) expected (%d,%d,%d)",
                 (int)dims[0], (int)dims[1], (int)dims[2],
                 CHESS_HALFKP_FEATURES, CHESS_NNUE_ACCUM, CHESS_NNUE_HIDDEN);
        fclose(f);
        return 0;
    }
    if (!read_exact(f, &search_depth, sizeof(search_depth))) {
        snprintf(err, err_sz, "qpolicy file too short (search_depth)");
        fclose(f);
        return 0;
    }
    if (search_depth < 1) search_depth = 1;
    if (search_depth > 8) search_depth = 8;
    g_qpol.search_depth = search_depth;

    if (!read_exact(f, g_qpol.in_bias, sizeof(g_qpol.in_bias)) ||
        !read_exact(f, g_qpol.in_w, sizeof(g_qpol.in_w)) ||
        !read_exact(f, g_qpol.l1_w, sizeof(g_qpol.l1_w)) ||
        !read_exact(f, g_qpol.l1_b, sizeof(g_qpol.l1_b)) ||
        !read_exact(f, g_qpol.l2_w, sizeof(g_qpol.l2_w)) ||
        !read_exact(f, g_qpol.l2_b, sizeof(g_qpol.l2_b)) ||
        !read_exact(f, g_qpol.out_w, sizeof(g_qpol.out_w)) ||
        !read_exact(f, &out_b, sizeof(out_b))) {
        snprintf(err, err_sz, "qpolicy file too short (weights)");
        fclose(f);
        qpol_clear();
        return 0;
    }
    g_qpol.out_b = out_b;

    fclose(f);
    g_qpol.loaded = 1;
    return 1;
}

/* ================================================================
 * Helper: load FEN strings from a file (one per line)
 * Returns number of FENs loaded, or -1 on error.
 * ================================================================ */
static int load_fen_file(const char* path, char*** out_list) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    int capacity = 256;
    int count = 0;
    char** list = (char**)malloc(capacity * sizeof(char*));
    if (!list) { fclose(f); return -1; }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        /* Strip trailing newline/carriage return */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;  /* skip empty lines */

        if (count >= capacity) {
            capacity *= 2;
            char** tmp = (char**)realloc(list, capacity * sizeof(char*));
            if (!tmp) { /* free what we have */
                for (int i = 0; i < count; i++) free(list[i]);
                free(list); fclose(f); return -1;
            }
            list = tmp;
        }
        list[count] = (char*)malloc(len + 1);
        if (!list[count]) {
            for (int i = 0; i < count; i++) free(list[i]);
            free(list); fclose(f); return -1;
        }
        memcpy(list[count], line, len + 1);
        count++;
    }
    fclose(f);
    *out_list = list;
    return count;
}

/* Free a FEN list */
static void free_fen_list(char** list, int count) {
    if (!list) return;
    for (int i = 0; i < count; i++) free(list[i]);
    free(list);
}

/* ================================================================
 * Helper: extract a double from kwargs by key
 * ================================================================ */
static double unpack_kwarg(PyObject* kwargs, const char* key, double fallback) {
    if (!kwargs) return fallback;
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (!val) return fallback;
    if (PyLong_Check(val)) return (double)PyLong_AsLong(val);
    if (PyFloat_Check(val)) return PyFloat_AsDouble(val);
    return fallback;
}

/* ================================================================
 * vec_init(observations, actions, rewards, terminals, truncations,
 *          num_agents, seed, *, max_steps=256, illegal_move_penalty=-0.1,
 *          num_games=num_agents/2)
 * ================================================================ */
static PyObject* vec_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError,
            "vec_init requires 7 positional args: obs, act, rew, term, trunc, num_agents, seed");
        return NULL;
    }

    /* Parse positional args */
    PyArrayObject* observations = (PyArrayObject*)PyTuple_GetItem(args, 0);
    PyArrayObject* actions      = (PyArrayObject*)PyTuple_GetItem(args, 1);
    PyArrayObject* rewards      = (PyArrayObject*)PyTuple_GetItem(args, 2);
    PyArrayObject* terminals    = (PyArrayObject*)PyTuple_GetItem(args, 3);
    /* truncations (index 4) unused but accepted for compatibility */

    int num_agents = (int)PyLong_AsLong(PyTuple_GetItem(args, 5));
    int seed       = (int)PyLong_AsLong(PyTuple_GetItem(args, 6));

    /* Parse kwargs */
    int max_steps           = (int)unpack_kwarg(kwargs, "max_steps", 256);
    float illegal_penalty   = (float)unpack_kwarg(kwargs, "illegal_move_penalty", -0.1);
    int num_games           = (int)unpack_kwarg(kwargs, "num_games", num_agents);
    float reward_invalid_piece = (float)unpack_kwarg(kwargs, "reward_invalid_piece", -0.01);
    float reward_invalid_move  = (float)unpack_kwarg(kwargs, "reward_invalid_move", -0.01);
    float reward_valid_piece   = (float)unpack_kwarg(kwargs, "reward_valid_piece", 0.0);
    float reward_valid_move    = (float)unpack_kwarg(kwargs, "reward_valid_move", 0.0);
    float reward_capture_bonus = (float)unpack_kwarg(kwargs, "reward_capture_bonus", 0.0);
    float reward_check_bonus   = (float)unpack_kwarg(kwargs, "reward_check_bonus", 0.0);
    float reward_repetition    = (float)unpack_kwarg(kwargs, "reward_repetition", 0.0);
    float reward_material      = (float)unpack_kwarg(kwargs, "reward_material", 0.0);
    float reward_position      = (float)unpack_kwarg(kwargs, "reward_position", 0.0);
    float reward_castling      = (float)unpack_kwarg(kwargs, "reward_castling", 0.0);
    float reward_draw          = (float)unpack_kwarg(kwargs, "reward_draw", 0.0);
    float reward_truncation    = (float)unpack_kwarg(kwargs, "reward_truncation", reward_draw);
    int enable_50_move_rule    = (int)unpack_kwarg(kwargs, "enable_50_move_rule", 1);
    int enable_threefold_repetition = (int)unpack_kwarg(kwargs, "enable_threefold_repetition", 1);
    int qpolicy_root_cap       = (int)unpack_kwarg(kwargs, "qpolicy_root_cap", CHESS_QPOL_ROOT_CAP_DEFAULT);
    float reward_see_hanging   = (float)unpack_kwarg(kwargs, "reward_see_hanging", 0.0);
    if (reward_see_hanging > 0.0f) reward_see_hanging = 0.0f;  // must be <= 0 (penalty)
    float fen_curric_pct       = (float)unpack_kwarg(kwargs, "fen_curric_pct", 0.0);

    /* Parse fen_file string kwarg */
    const char* fen_file_path = NULL;
    if (kwargs) {
        PyObject* fen_val = PyDict_GetItemString(kwargs, "fen_file");
        if (fen_val && PyUnicode_Check(fen_val)) {
            fen_file_path = PyUnicode_AsUTF8(fen_val);
        }
    }

    if (num_agents != num_games) {
        PyErr_SetString(PyExc_ValueError,
            "num_agents must equal num_games (1 agent per game)");
        return NULL;
    }

    /* Validate numpy arrays */
    if (!PyArray_ISCONTIGUOUS(observations) || !PyArray_ISCONTIGUOUS(actions) ||
        !PyArray_ISCONTIGUOUS(rewards) || !PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "All arrays must be contiguous");
        return NULL;
    }
    if (PyArray_TYPE(observations) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "observations array must be float32");
        return NULL;
    }

    /* Allocate VecEnv */
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate VecEnv");
        return NULL;
    }
    vec->num_games  = num_games;
    vec->num_agents = num_agents;
    vec->fen_list = NULL;
    vec->fen_count = 0;
    vec->fen_curric_pct = fen_curric_pct;
    vec->games      = (ChessEnv*)calloc(num_games, sizeof(ChessEnv));
    if (!vec->games) {
        free(vec);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate ChessEnv array");
        return NULL;
    }

    /* Load FEN file if provided */
    if (fen_file_path && fen_file_path[0]) {
        int count = load_fen_file(fen_file_path, &vec->fen_list);
        if (count < 0) {
            free(vec->games); free(vec);
            PyErr_Format(PyExc_IOError, "Failed to load FEN file: %s", fen_file_path);
            return NULL;
        }
        vec->fen_count = count;
    }

    /* Get raw data pointers */
    float* obs_data          = (float*)PyArray_DATA(observations);
    int obs_stride_bytes     = (int)PyArray_STRIDE(observations, 0); /* bytes per agent obs row */

    /* actions must be int32 or int64 */
    void* act_data           = PyArray_DATA(actions);
    int act_itemsize         = (int)PyArray_ITEMSIZE(actions);
    if ((act_itemsize != 4 && act_itemsize != 8) || !PyArray_ISINTEGER(actions)) {
        free(vec->games); free(vec);
        PyErr_SetString(PyExc_TypeError,
            "actions array must be integer dtype (int32 or int64)");
        return NULL;
    }

    float* rew_data          = (float*)PyArray_DATA(rewards);
    unsigned char* term_data = (unsigned char*)PyArray_DATA(terminals);

    /* Initialize bitboard tables (once) */
    if (!bitboards_initialized) init_bitboards();

    /* Wire up each game: 1 agent per game */
    for (int g = 0; g < num_games; g++) {
        ChessEnv* game = &vec->games[g];

        game->observations = (float*)((char*)obs_data + g * obs_stride_bytes);
        game->obs_stride   = 0;  /* unused in 1-agent mode */

        game->actions        = (char*)act_data + g * act_itemsize;
        game->action_itemsize = act_itemsize;

        game->rewards   = rew_data + g;
        game->terminals = term_data + g;

        /* Alternate initial learner_color across games */
        game->learner_color = g % 2;

        game->rng_state = (uint64_t)(seed * num_games + g + 1);
        if (game->rng_state == 0) game->rng_state = 1;

        /* init() sets defaults; call BEFORE overriding with user params */
        init(game);
        game->max_steps = max_steps;
        game->illegal_move_penalty = illegal_penalty;
        game->reward_invalid_piece = reward_invalid_piece;
        game->reward_invalid_move = reward_invalid_move;
        game->reward_valid_piece = reward_valid_piece;
        game->reward_valid_move = reward_valid_move;
        game->reward_capture_bonus = reward_capture_bonus;
        game->reward_check_bonus = reward_check_bonus;
        game->reward_repetition = reward_repetition;
        game->reward_material = reward_material;
        game->reward_position = reward_position;
        game->reward_castling = reward_castling;
        game->reward_draw = reward_draw;
        game->reward_truncation = reward_truncation;
        game->enable_50_move_rule = enable_50_move_rule;
        game->enable_threefold_repetition = enable_threefold_repetition;
        game->qpol_root_cap = qpolicy_root_cap;
        game->reward_see_hanging = reward_see_hanging;
        game->use_curriculum = (fen_curric_pct > 0.0f && vec->fen_count > 0) ? 1 : 0;
        game->obs_stride = obs_stride_bytes;
    }

    return PyLong_FromVoidPtr(vec);
}

/* ================================================================
 * vec_reset(handle, seed)
 * ================================================================ */
/* Reset a single game, optionally using FEN curriculum */
static void vec_reset_game(VecEnv* vec, int g) {
    ChessEnv* game = &vec->games[g];
    c_reset(game);

    /* FEN curriculum: with probability fen_curric_pct, use a random FEN */
    if (game->use_curriculum && vec->fen_count > 0 && vec->fen_curric_pct > 0.0f) {
        /* Generate random float in [0,1) */
        float r = (float)(chess_xorshift64(&game->rng_state) & 0xFFFFFF) / (float)0x1000000;
        if (r < vec->fen_curric_pct) {
            int idx = chess_rand_int(&game->rng_state, vec->fen_count);
            setup_from_fen(game, vec->fen_list[idx]);
            /* Align learner perspective with side-to-move on curriculum starts. */
            game->learner_color = game->current_player;
            /* Re-reset phase state and write obs for the FEN position */
            game->step_count = 0;
            game->phase_state[0].pick_phase = 0;
            game->phase_state[0].selected_square = -1;
            game->phase_state[0].valid_dest_count = 0;
            game->phase_state[0].planned_valid = 0;
            game->phase_state[0].planned_from_sq = -1;
            game->phase_state[0].planned_to_sq = -1;
            game->phase_state[0].planned_phase1_action = 0;
            game->rewards[0] = 0.0f;
            game->terminals[0] = 0;
            /* Re-record position hash for the FEN position */
            game->position_history_count = 0;
            record_position_hash(game);
            rebuild_halfkp_accumulator(game);
            write_observations(game);
        }
    }
}

static PyObject* vec_reset(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    int seed = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));

    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < vec->num_games; g++) {
        vec->games[g].rng_state = (uint64_t)(seed * vec->num_games + g + 1);
        if (vec->games[g].rng_state == 0) vec->games[g].rng_state = 1;
        /* Zero the log on full reset (vec_reset) */
        memset(&vec->games[g].log, 0, sizeof(Log));
        vec_reset_game(vec, g);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ================================================================
 * vec_step(handle)
 * ================================================================ */
static PyObject* vec_step(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));

    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < vec->num_games; g++) {
        ChessEnv* game = &vec->games[g];
        int was_terminal = game->terminals[0];
        c_step(game);
        /* If c_step just auto-reset (was terminal), apply FEN curriculum */
        if (was_terminal && game->use_curriculum && vec->fen_count > 0 && vec->fen_curric_pct > 0.0f) {
            float r = (float)(chess_xorshift64(&game->rng_state) & 0xFFFFFF) / (float)0x1000000;
            if (r < vec->fen_curric_pct) {
                int idx = chess_rand_int(&game->rng_state, vec->fen_count);
                setup_from_fen(game, vec->fen_list[idx]);
                /* Align learner perspective with side-to-move on curriculum starts. */
                game->learner_color = game->current_player;
                game->step_count = 0;
                game->phase_state[0].pick_phase = 0;
                game->phase_state[0].selected_square = -1;
                game->phase_state[0].valid_dest_count = 0;
                game->phase_state[0].planned_valid = 0;
                game->phase_state[0].planned_from_sq = -1;
                game->phase_state[0].planned_to_sq = -1;
                game->phase_state[0].planned_phase1_action = 0;
                game->rewards[0] = 0.0f;
                game->terminals[0] = 0;
                /* Re-record position hash for the FEN position */
                game->position_history_count = 0;
                record_position_hash(game);
                rebuild_halfkp_accumulator(game);
                write_observations(game);
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ================================================================
 * vec_log(handle) -> dict
 *
 * Aggregates logs across all games, resets per-game logs.
 * Returns averaged values.
 * ================================================================ */
static PyObject* assign_float(PyObject* dict, const char* key, float val) {
    PyObject* v = PyFloat_FromDouble((double)val);
    if (!v) return NULL;
    PyDict_SetItemString(dict, key, v);
    Py_DECREF(v);
    return dict;
}

static PyObject* vec_log(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));

    /* Aggregate logs */
    Log agg;
    memset(&agg, 0, sizeof(Log));

    int num_keys = (int)(sizeof(Log) / sizeof(float));
    for (int g = 0; g < vec->num_games; g++) {
        Log* lg = &vec->games[g].log;
        float* src = (float*)lg;
        float* dst = (float*)&agg;
        for (int k = 0; k < num_keys; k++) {
            dst[k] += src[k];
            src[k] = 0.0f;
        }
    }

    PyObject* dict = PyDict_New();
    if (agg.n == 0.0f) return dict;

    /* Average by number of completed episodes */
    float n = agg.n;
    float inv_n = 1.0f / n;
    agg.episode_length *= inv_n;
    agg.episode_return *= inv_n;
    agg.white_wins     *= inv_n;
    agg.black_wins     *= inv_n;
    agg.draws          *= inv_n;
    agg.illegal_moves  *= inv_n;
    agg.material_score *= inv_n;
    agg.positional_score *= inv_n;
    agg.invalid_action_rate *= inv_n;
    agg.chess_moves    *= inv_n;
    agg.repetitions    *= inv_n;

    /* Compute derived fields from raw aggregated values */
    float draw_rate = agg.draws;
    float white_winrate = agg.white_wins;
    float black_winrate = agg.black_wins;
    float score = white_winrate + 0.5f * draw_rate;

    assign_float(dict, "episode_length", agg.episode_length);
    assign_float(dict, "episode_return", agg.episode_return);
    assign_float(dict, "white_wins", agg.white_wins);
    assign_float(dict, "black_wins", agg.black_wins);
    assign_float(dict, "draws", agg.draws);
    assign_float(dict, "illegal_moves", agg.illegal_moves);
    assign_float(dict, "material_score", agg.material_score);
    assign_float(dict, "positional_score", agg.positional_score);
    assign_float(dict, "invalid_action_rate", agg.invalid_action_rate);
    assign_float(dict, "draw_rate", draw_rate);
    assign_float(dict, "white_winrate", white_winrate);
    assign_float(dict, "black_winrate", black_winrate);
    assign_float(dict, "score", score);
    assign_float(dict, "chess_moves", agg.chess_moves);
    assign_float(dict, "repetitions", agg.repetitions);
    assign_float(dict, "n", n);

    return dict;
}

/* ================================================================
 * vec_load_fens(handle, fen_file_path)
 *
 * Load FENs after init. Replaces any previously loaded FEN list.
 * ================================================================ */
static PyObject* vec_load_fens(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    PyObject* path_obj = PyTuple_GetItem(args, 1);
    if (!PyUnicode_Check(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "fen_file_path must be a string");
        return NULL;
    }
    const char* path = PyUnicode_AsUTF8(path_obj);

    /* Free old FEN list if any */
    free_fen_list(vec->fen_list, vec->fen_count);
    vec->fen_list = NULL;
    vec->fen_count = 0;

    char** new_list = NULL;
    int count = load_fen_file(path, &new_list);
    if (count < 0) {
        PyErr_Format(PyExc_IOError, "Failed to load FEN file: %s", path);
        return NULL;
    }
    vec->fen_list = new_list;
    vec->fen_count = count;

    /* Update use_curriculum flag on all games */
    for (int g = 0; g < vec->num_games; g++) {
        vec->games[g].use_curriculum = (vec->fen_curric_pct > 0.0f && count > 0) ? 1 : 0;
    }

    return PyLong_FromLong(count);
}

/* ================================================================
 * vec_set_fen_pct(handle, pct)
 *
 * Update the FEN curriculum percentage at runtime.
 * ================================================================ */
static PyObject* vec_set_fen_pct(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    float pct = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
    vec->fen_curric_pct = pct;
    for (int g = 0; g < vec->num_games; g++) {
        vec->games[g].use_curriculum = (pct > 0.0f && vec->fen_count > 0) ? 1 : 0;
    }
    Py_RETURN_NONE;
}

/* ================================================================
 * vec_load_qpolicy(handle, qpolicy_path)
 * ================================================================ */
static PyObject* vec_load_qpolicy(PyObject* self, PyObject* args) {
    (void)self;
    (void)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));  // keep API symmetric with vec_* handle methods
    PyObject* path_obj = PyTuple_GetItem(args, 1);
    if (!PyUnicode_Check(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "qpolicy_path must be a string");
        return NULL;
    }
    const char* path = PyUnicode_AsUTF8(path_obj);
    char err[256];
    if (!load_qpolicy_file(path, err, sizeof(err))) {
        PyErr_SetString(PyExc_IOError, err);
        return NULL;
    }
    Py_RETURN_TRUE;
}

/* ================================================================
 * vec_infer_actions(handle, out_actions[, out_values])
 * ================================================================ */
static PyObject* vec_infer_actions(PyObject* self, PyObject* args) {
    (void)self;
    int nargs = (int)PyTuple_Size(args);
    if (nargs < 2 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "vec_infer_actions expects (handle, out_actions[, out_values])");
        return NULL;
    }
    if (!qpol_is_loaded()) {
        PyErr_SetString(PyExc_RuntimeError, "NNUE policy not loaded. Call vec_load_qpolicy first.");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    PyArrayObject* out_actions = (PyArrayObject*)PyTuple_GetItem(args, 1);
    if (!PyArray_ISCONTIGUOUS(out_actions) || PyArray_TYPE(out_actions) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "out_actions must be contiguous int32 array");
        return NULL;
    }
    if (PyArray_SIZE(out_actions) < vec->num_games) {
        PyErr_SetString(PyExc_ValueError, "out_actions size is smaller than num_games");
        return NULL;
    }
    int32_t* actions = (int32_t*)PyArray_DATA(out_actions);

    float* values = NULL;
    if (nargs == 3) {
        PyArrayObject* out_values = (PyArrayObject*)PyTuple_GetItem(args, 2);
        if (!PyArray_ISCONTIGUOUS(out_values) || PyArray_TYPE(out_values) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_TypeError, "out_values must be contiguous float32 array");
            return NULL;
        }
        if (PyArray_SIZE(out_values) < vec->num_games) {
            PyErr_SetString(PyExc_ValueError, "out_values size is smaller than num_games");
            return NULL;
        }
        values = (float*)PyArray_DATA(out_values);
    }

    for (int g = 0; g < vec->num_games; g++) {
        float v = 0.0f;
        int action = qpol_select_action(&vec->games[g], &v);
        actions[g] = (int32_t)action;
        if (values) values[g] = v;
    }
    return PyLong_FromLong(vec->num_games);
}

/* ================================================================
 * vec_step_qpolicy(handle)
 *
 * Fully C-side inference + stepping. No Python policy forward call.
 * ================================================================ */
static PyObject* vec_step_qpolicy(PyObject* self, PyObject* args) {
    (void)self;
    if (!qpol_is_loaded()) {
        PyErr_SetString(PyExc_RuntimeError, "NNUE policy not loaded. Call vec_load_qpolicy first.");
        return NULL;
    }
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));

    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < vec->num_games; g++) {
        ChessEnv* game = &vec->games[g];
        int was_terminal = game->terminals[0];
        if (!was_terminal) {
            int action = qpol_select_action(game, NULL);
            set_action(game, 0, action);
        }
        c_step(game);

        if (was_terminal && game->use_curriculum && vec->fen_count > 0 && vec->fen_curric_pct > 0.0f) {
            float r = (float)(chess_xorshift64(&game->rng_state) & 0xFFFFFF) / (float)0x1000000;
            if (r < vec->fen_curric_pct) {
                int idx = chess_rand_int(&game->rng_state, vec->fen_count);
                setup_from_fen(game, vec->fen_list[idx]);
                game->learner_color = game->current_player;
                game->step_count = 0;
                game->phase_state[0].pick_phase = 0;
                game->phase_state[0].selected_square = -1;
                game->phase_state[0].valid_dest_count = 0;
                game->phase_state[0].planned_valid = 0;
                game->phase_state[0].planned_from_sq = -1;
                game->phase_state[0].planned_to_sq = -1;
                game->phase_state[0].planned_phase1_action = 0;
                game->rewards[0] = 0.0f;
                game->terminals[0] = 0;
                game->position_history_count = 0;
                record_position_hash(game);
                rebuild_halfkp_accumulator(game);
                write_observations(game);
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ================================================================
 * vec_close(handle)
 * ================================================================ */
static PyObject* vec_close(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (vec) {
        for (int g = 0; g < vec->num_games; g++) {
            c_close(&vec->games[g]);
        }
        free(vec->games);
        free_fen_list(vec->fen_list, vec->fen_count);
        free(vec);
    }
    Py_RETURN_NONE;
}

/* ================================================================
 * Module definition
 * ================================================================ */
static PyMethodDef methods[] = {
    {"vec_init",      (PyCFunction)vec_init,  METH_VARARGS | METH_KEYWORDS, "Init vectorized chess environments"},
    {"vec_reset",     vec_reset,     METH_VARARGS, "Reset all games"},
    {"vec_step",      vec_step,      METH_VARARGS, "Step all games"},
    {"vec_log",       vec_log,       METH_VARARGS, "Aggregate and return logs"},
    {"vec_load_fens", vec_load_fens, METH_VARARGS, "Load FEN file for curriculum"},
    {"vec_set_fen_pct", vec_set_fen_pct, METH_VARARGS, "Set FEN curriculum percentage"},
    {"vec_load_qpolicy", vec_load_qpolicy, METH_VARARGS, "Load native NNUE policy binary"},
    {"vec_infer_actions", vec_infer_actions, METH_VARARGS, "Infer actions (and optional values) using NNUE search"},
    {"vec_step_qpolicy", vec_step_qpolicy, METH_VARARGS, "Infer and step fully in C using NNUE search"},
    {"vec_close",     vec_close,     METH_VARARGS, "Free all games"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "binding",
    "Chess self-play C binding for PufferLib",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    return PyModule_Create(&module);
}
