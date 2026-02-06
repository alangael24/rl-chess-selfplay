/*
 * binding.c - Custom PufferLib C binding for Chess self-play.
 *
 * Unlike the standard env_binding.h pattern (one Env per agent),
 * chess.h uses one ChessEnv per GAME that manages 2 agent slots.
 *
 * Layout in PufferLib buffers (for N games, 2N agent slots):
 *   observations[2*i]   = White obs for game i
 *   observations[2*i+1] = Black obs for game i
 *   actions[2*i]        = White action for game i
 *   actions[2*i+1]      = Black action for game i
 *   rewards[2*i]        = White reward for game i
 *   rewards[2*i+1]      = Black reward for game i
 *   terminals[2*i]      = White terminal for game i
 *   terminals[2*i+1]    = Black terminal for game i
 *
 * ChessEnv.observations -> &observations[2*i * OBS_SIZE]
 * ChessEnv.actions      -> &actions[2*i]
 * ChessEnv.rewards      -> &rewards[2*i]
 * ChessEnv.terminals    -> &terminals[2*i]
 * ChessEnv.obs_stride   = OBS_SIZE (to get from white obs to black obs)
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include "chess.h"

typedef struct {
    ChessEnv* games;
    int num_games;
    int num_agents;  /* = num_games * 2 */
} VecEnv;

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
    int num_games           = (int)unpack_kwarg(kwargs, "num_games", num_agents / 2);

    if (num_agents != num_games * 2) {
        PyErr_SetString(PyExc_ValueError,
            "num_agents must equal num_games * 2");
        return NULL;
    }

    /* Validate numpy arrays */
    if (!PyArray_ISCONTIGUOUS(observations) || !PyArray_ISCONTIGUOUS(actions) ||
        !PyArray_ISCONTIGUOUS(rewards) || !PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "All arrays must be contiguous");
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
    vec->games      = (ChessEnv*)calloc(num_games, sizeof(ChessEnv));
    if (!vec->games) {
        free(vec);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate ChessEnv array");
        return NULL;
    }

    /* Get raw data pointers */
    unsigned char* obs_data  = (unsigned char*)PyArray_DATA(observations);
    int obs_stride_bytes     = (int)PyArray_STRIDE(observations, 0); /* bytes per agent obs row */

    /* actions could be int32 or int64 depending on PufferLib version */
    void* act_data           = PyArray_DATA(actions);
    int act_itemsize         = (int)PyArray_ITEMSIZE(actions);

    float* rew_data          = (float*)PyArray_DATA(rewards);
    unsigned char* term_data = (unsigned char*)PyArray_DATA(terminals);

    /* Wire up each game */
    for (int g = 0; g < num_games; g++) {
        ChessEnv* game = &vec->games[g];

        /* White = agent 2*g, Black = agent 2*g+1 */
        int white_idx = 2 * g;

        game->observations = obs_data + white_idx * obs_stride_bytes;
        game->obs_stride   = obs_stride_bytes;  /* distance from white obs to black obs */

        /* actions: point to the white agent slot; game reads [0] for white, [1] for black.
         * get_action() in chess.h handles int32 vs int64 via action_itemsize. */
        game->actions        = (char*)act_data + white_idx * act_itemsize;
        game->action_itemsize = act_itemsize;

        game->rewards   = rew_data + white_idx;
        game->terminals = term_data + white_idx;

        game->rng_state = (uint64_t)(seed * num_games + g + 1);
        if (game->rng_state == 0) game->rng_state = 1;

        /* init() sets defaults; call BEFORE overriding with user params */
        init(game);
        game->max_steps = max_steps;
        game->illegal_move_penalty = illegal_penalty;
        game->obs_stride = obs_stride_bytes;
    }

    return PyLong_FromVoidPtr(vec);
}

/* ================================================================
 * vec_reset(handle, seed)
 * ================================================================ */
static PyObject* vec_reset(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    int seed = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));

    for (int g = 0; g < vec->num_games; g++) {
        vec->games[g].rng_state = (uint64_t)(seed * vec->num_games + g + 1);
        if (vec->games[g].rng_state == 0) vec->games[g].rng_state = 1;
        c_reset(&vec->games[g]);
    }
    Py_RETURN_NONE;
}

/* ================================================================
 * vec_step(handle)
 * ================================================================ */
static PyObject* vec_step(PyObject* self, PyObject* args) {
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));

    for (int g = 0; g < vec->num_games; g++) {
        c_step(&vec->games[g]);
    }
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

    assign_float(dict, "episode_length", agg.episode_length);
    assign_float(dict, "episode_return", agg.episode_return);
    assign_float(dict, "white_wins", agg.white_wins);
    assign_float(dict, "black_wins", agg.black_wins);
    assign_float(dict, "draws", agg.draws);
    assign_float(dict, "illegal_moves", agg.illegal_moves);
    assign_float(dict, "n", n);

    return dict;
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
        free(vec);
    }
    Py_RETURN_NONE;
}

/* ================================================================
 * Module definition
 * ================================================================ */
static PyMethodDef methods[] = {
    {"vec_init",  (PyCFunction)vec_init,  METH_VARARGS | METH_KEYWORDS, "Init vectorized chess environments"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset all games"},
    {"vec_step",  vec_step,  METH_VARARGS, "Step all games"},
    {"vec_log",   vec_log,   METH_VARARGS, "Aggregate and return logs"},
    {"vec_close", vec_close, METH_VARARGS, "Free all games"},
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
