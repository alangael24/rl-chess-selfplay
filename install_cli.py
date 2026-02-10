"""Install Chess environment into PufferLib's Ocean system for CLI support.

After running this script, you can use:
    puffer train puffer_chess
    python -m pufferlib.pufferl train puffer_chess

Usage:
    python install_cli.py          # Install + build
    python install_cli.py --clean  # Remove chess from pufferlib
"""

import os
import sys
import shutil
import subprocess
import importlib

def get_pufferlib_dir():
    import pufferlib
    return os.path.dirname(pufferlib.__file__)

def install():
    here = os.path.dirname(os.path.abspath(__file__))
    pufferlib_dir = get_pufferlib_dir()
    ocean_dir = os.path.join(pufferlib_dir, 'ocean')
    chess_ocean_dir = os.path.join(ocean_dir, 'chess')
    config_ocean_dir = os.path.join(pufferlib_dir, 'config', 'ocean')

    print(f"PufferLib dir: {pufferlib_dir}")
    print(f"Chess ocean dir: {chess_ocean_dir}")

    # 1. Create ocean/chess/ directory
    os.makedirs(chess_ocean_dir, exist_ok=True)

    # 2. Copy C source files
    for fname in ['chess.h', 'binding.c']:
        src = os.path.join(here, 'csrc', fname)
        dst = os.path.join(chess_ocean_dir, fname)
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

    # 3. Create chess.py (ocean-compatible version)
    chess_py_path = os.path.join(chess_ocean_dir, 'chess.py')
    with open(chess_py_path, 'w') as f:
        f.write(CHESS_PY_CONTENT)
    print("  Created chess.py")

    # 4. Create __init__.py
    init_path = os.path.join(chess_ocean_dir, '__init__.py')
    with open(init_path, 'w') as f:
        f.write('')
    print("  Created __init__.py")

    # 5. Build the C extension in-place
    print("\nBuilding C extension...")
    import numpy as np
    compile_args = ['-O3', '-ffast-math', '-march=native', '-std=c11']
    link_args = []
    if sys.platform == 'darwin':
        compile_args.append('-stdlib=libc++')
    elif sys.platform.startswith('linux'):
        compile_args.append('-fopenmp')
        link_args.append('-fopenmp')

    binding_c = os.path.join(chess_ocean_dir, 'binding.c')
    import sysconfig
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    binding_so = os.path.join(chess_ocean_dir, f'binding{ext_suffix}')

    include_dirs = [
        chess_ocean_dir,
        np.get_include(),
        ocean_dir,
    ]
    include_flags = [f'-I{d}' for d in include_dirs]

    # Get Python include and lib flags
    python_include = sysconfig.get_path('include')
    python_flags = [f'-I{python_include}']

    # Get linker flags
    python_lib = sysconfig.get_config_var('LIBDIR')
    python_ldflags = [f'-L{python_lib}'] if python_lib else []
    python_ver = sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION')

    cmd = [
        'gcc', '-shared', '-fPIC',
        *compile_args,
        *include_flags,
        *python_flags,
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        binding_c,
        '-o', binding_so,
        *link_args,
        *python_ldflags,
        f'-lpython{python_ver}',
    ]

    # Try gcc first, fall back to setuptools
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Built {os.path.basename(binding_so)}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Direct gcc failed, using setuptools...")
        build_with_setuptools(chess_ocean_dir, ocean_dir)

    # 6. Add chess to MAKE_FUNCTIONS in environment.py
    env_py = os.path.join(ocean_dir, 'environment.py')
    with open(env_py, 'r') as f:
        content = f.read()

    if "'chess'" not in content:
        content = content.replace(
            "MAKE_FUNCTIONS = {",
            "MAKE_FUNCTIONS = {\n    'chess': 'Chess',",
        )
        with open(env_py, 'w') as f:
            f.write(content)
        print("  Registered chess in MAKE_FUNCTIONS")
    else:
        print("  chess already in MAKE_FUNCTIONS")

    # 7. Add chess config .ini
    config_path = os.path.join(config_ocean_dir, 'chess.ini')
    with open(config_path, 'w') as f:
        f.write(CHESS_INI_CONTENT)
    print(f"  Created config: {config_path}")

    # 8. Add/update Policy in ocean/torch.py
    torch_py = os.path.join(ocean_dir, 'torch.py')
    with open(torch_py, 'r') as f:
        torch_content = f.read()

    marker = '\n# === Chess Self-Play Policy ==='
    if marker in torch_content:
        torch_content = torch_content[:torch_content.index(marker)]
    torch_content += CHESS_POLICY_CONTENT
    with open(torch_py, 'w') as f:
        f.write(torch_content)
    print("  Updated Chess policy in torch.py")

    # Clear Python caches
    for root, dirs, files in os.walk(ocean_dir):
        for d in dirs:
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print("\n" + "=" * 50)
    print("Chess environment installed in PufferLib!")
    print("Usage:")
    print("  puffer train puffer_chess")
    print("  python -m pufferlib.pufferl train puffer_chess")
    print("=" * 50)


def build_with_setuptools(chess_ocean_dir, ocean_dir):
    """Fallback: build using setuptools."""
    import numpy as np
    openmp_compile = '["-fopenmp"]' if sys.platform.startswith('linux') else '[]'
    openmp_link = '["-fopenmp"]' if sys.platform.startswith('linux') else '[]'
    setup_content = f'''
from setuptools import setup, Extension
import numpy as np
import sys
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-std=c11"]
extra_link_args = []
if sys.platform.startswith("linux"):
    extra_compile_args += {openmp_compile}
    extra_link_args += {openmp_link}
setup(
    name="chess-binding",
    ext_modules=[Extension(
        "binding",
        sources=["{os.path.join(chess_ocean_dir, "binding.c")}"],
        include_dirs=["{chess_ocean_dir}", "{np.get_include()}", "{ocean_dir}"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )],
)
'''
    setup_path = os.path.join(chess_ocean_dir, '_setup_tmp.py')
    with open(setup_path, 'w') as f:
        f.write(setup_content)

    subprocess.run(
        [sys.executable, setup_path, 'build_ext', '--inplace'],
        cwd=chess_ocean_dir,
        check=True,
    )
    os.remove(setup_path)
    # Clean build artifacts
    build_dir = os.path.join(chess_ocean_dir, 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)


def clean():
    pufferlib_dir = get_pufferlib_dir()
    ocean_dir = os.path.join(pufferlib_dir, 'ocean')
    chess_ocean_dir = os.path.join(ocean_dir, 'chess')
    config_path = os.path.join(pufferlib_dir, 'config', 'ocean', 'chess.ini')

    # Remove chess ocean dir
    if os.path.exists(chess_ocean_dir):
        shutil.rmtree(chess_ocean_dir)
        print(f"  Removed {chess_ocean_dir}")

    # Remove config
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"  Removed {config_path}")

    # Remove chess from MAKE_FUNCTIONS
    env_py = os.path.join(ocean_dir, 'environment.py')
    with open(env_py, 'r') as f:
        content = f.read()
    if "'chess': 'Chess'," in content:
        content = content.replace("    'chess': 'Chess',\n", "")
        with open(env_py, 'w') as f:
            f.write(content)
        print("  Removed chess from MAKE_FUNCTIONS")

    # Remove Policy from torch.py
    torch_py = os.path.join(ocean_dir, 'torch.py')
    with open(torch_py, 'r') as f:
        torch_content = f.read()
    marker = '\n# === Chess Self-Play Policy ==='
    if marker in torch_content:
        torch_content = torch_content[:torch_content.index(marker)]
        with open(torch_py, 'w') as f:
            f.write(torch_content)
        print("  Removed Chess policy from torch.py")

    print("Chess environment removed from PufferLib.")


# --- Ocean-compatible chess.py ---

CHESS_PY_CONTENT = '''\
"""PufferLib Ocean Chess self-play environment.

1-agent-per-game: each step the agent controls whoever's turn it is.
Two-phase action system: 97 actions (pick piece, pick dest, pass[legacy]).
"""

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

ACCUM_SIZE = 256
OBS_META = 3
OBS_SIZE = ACCUM_SIZE + OBS_META
NUM_ACTIONS = 97  # 64 squares + 32 promotions + 1 pass


class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=128, render_mode=None, report_interval=128,
                 max_steps=256, illegal_move_penalty=-0.1,
                 reward_invalid_piece=-0.01, reward_invalid_move=-0.01,
                 reward_valid_piece=0.0, reward_valid_move=0.0,
                 reward_capture_bonus=0.0, reward_check_bonus=0.0,
                 reward_repetition=0.0, reward_material=0.0,
                 reward_position=0.0, reward_castling=0.0,
                 reward_draw=0.0, reward_see_hanging=0.0,
                 enable_50_move_rule=1,
                 enable_threefold_repetition=1,
                 use_native_qpolicy=0, qpolicy_path=None, qpolicy_root_cap=12,
                 fen_file=None, fen_curric_pct=0.0,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        # Required by newer PufferLib buffer setup path.
        self.selfplay = 0
        self.num_agents = num_envs  # 1 agent per game
        self.agents_per_batch = self.num_agents

        super().__init__(buf=buf)

        init_kwargs = dict(
            max_steps=max_steps,
            illegal_move_penalty=illegal_move_penalty,
            reward_invalid_piece=reward_invalid_piece,
            reward_invalid_move=reward_invalid_move,
            reward_valid_piece=reward_valid_piece,
            reward_valid_move=reward_valid_move,
            reward_capture_bonus=reward_capture_bonus,
            reward_check_bonus=reward_check_bonus,
            reward_repetition=reward_repetition,
            reward_material=reward_material,
            reward_position=reward_position,
            reward_castling=reward_castling,
            reward_draw=reward_draw,
            reward_see_hanging=reward_see_hanging,
            enable_50_move_rule=enable_50_move_rule,
            enable_threefold_repetition=enable_threefold_repetition,
            qpolicy_root_cap=qpolicy_root_cap,
            fen_curric_pct=fen_curric_pct,
            num_games=num_envs,
        )
        if fen_file is not None:
            init_kwargs['fen_file'] = fen_file
        if qpolicy_path is not None and str(qpolicy_path).strip() == '':
            qpolicy_path = None

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents, seed,
            **init_kwargs,
        )
        self.fen_curric_pct = float(fen_curric_pct)
        self.fen_file = fen_file
        self.use_native_qpolicy = bool(int(use_native_qpolicy))
        self.qpolicy_path = qpolicy_path

        if self.use_native_qpolicy:
            if self.qpolicy_path is None:
                raise ValueError('use_native_qpolicy=1 requires env.qpolicy_path')
            if not bool(binding.vec_load_qpolicy(self.c_envs, self.qpolicy_path)):
                raise RuntimeError(f'Failed to load qpolicy: {self.qpolicy_path}')

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
        return self.observations, []

    def step(self, actions):
        if self.use_native_qpolicy:
            binding.vec_step_qpolicy(self.c_envs)
        else:
            self.actions[:] = actions
            binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get('episode_length', 0) > 0:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)

    def load_fens(self, fen_file):
        loaded = int(binding.vec_load_fens(self.c_envs, fen_file))
        self.fen_file = fen_file
        return loaded

    def set_fen_curric_pct(self, pct):
        pct = float(max(0.0, min(1.0, pct)))
        binding.vec_set_fen_pct(self.c_envs, pct)
        self.fen_curric_pct = pct

    def load_qpolicy(self, qpolicy_path):
        self.qpolicy_path = qpolicy_path
        return bool(binding.vec_load_qpolicy(self.c_envs, qpolicy_path))
'''

# --- Config .ini ---

CHESS_INI_CONTENT = '''\
[base]
package = ocean
env_name = puffer_chess
policy_name = Chess
rnn_name = ChessLSTM

[env]
num_envs = 1024
max_steps = 256
illegal_move_penalty = -0.1
reward_invalid_piece = -0.01
reward_invalid_move = -0.01
reward_valid_piece = 0.0
reward_valid_move = 0.0
reward_capture_bonus = 0.0
reward_check_bonus = 0.0
reward_repetition = -0.01
reward_material = 0.0
reward_position = 0.0
reward_castling = 0.0
reward_draw = -0.02
reward_see_hanging = 0.0
enable_50_move_rule = 1
enable_threefold_repetition = 1
use_native_qpolicy = 0
qpolicy_path =
qpolicy_root_cap = 12
fen_file =
fen_curric_pct = 0.0
report_interval = 128

[vec]
backend = PufferEnv
num_envs = 1

[policy]
hidden_size = 256

[rnn]
input_size = 256
hidden_size = 256

[train]
total_timesteps = 1_000_000_000
precision = bfloat16
use_rnn = False
learning_rate = 1e-4
ent_coef = 0.005
gamma = 0.997
gae_lambda = 0.95
clip_coef = 0.15
vf_coef = 0.5
max_grad_norm = 1.0
update_epochs = 1
batch_size = 262144
minibatch_size = 65536
bptt_horizon = 128
checkpoint_interval = 500
anneal_lr = True
'''

# --- Policy class (appended to ocean/torch.py) ---

CHESS_POLICY_CONTENT = '''

# === Chess Self-Play Policy ===
CHESS_ACCUM_SIZE = 256
CHESS_OBS_PHASE = CHESS_ACCUM_SIZE
CHESS_OBS_LEARNER_TURN = CHESS_ACCUM_SIZE + 2
CHESS_NUM_ACTIONS = 97

class Chess(nn.Module):
    def __init__(self, env, hidden_size=256, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        in_features = CHESS_ACCUM_SIZE + 3
        self.backbone = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(in_features, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, CHESS_NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

        self._phase0 = None

    def encode_observations(self, x, state=None):
        accum = x[:, :CHESS_ACCUM_SIZE].float()
        phase = x[:, CHESS_OBS_PHASE:CHESS_OBS_PHASE + 2].float()
        learner_turn = x[:, CHESS_OBS_LEARNER_TURN:CHESS_OBS_LEARNER_TURN + 1].float()
        model_in = torch.cat([accum, phase, learner_turn], dim=1)
        hidden = self.backbone(model_in)
        self._phase0 = phase[:, 0:1] > 0.5
        return hidden

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        value = self.critic(hidden)

        if self._phase0 is not None and self._phase0.shape[0] == logits.shape[0]:
            logits = logits.clone()
            logits[:, 96] = -1e9
            promo_logits = logits[:, 64:96]
            phase0 = self._phase0.expand_as(promo_logits)
            logits[:, 64:96] = torch.where(
                phase0,
                torch.full_like(promo_logits, -1e9),
                promo_logits,
            )

        return logits, value

    def forward_eval(self, x, state=None):
        hidden = self.encode_observations(x, state=state)
        logits, value = self.decode_actions(hidden)
        return logits, value

    def forward(self, x, state=None):
        return self.forward_eval(x, state)


class ChessLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256):
        super().__init__(env, policy, input_size, hidden_size)
'''


if __name__ == '__main__':
    if '--clean' in sys.argv:
        clean()
    else:
        install()
