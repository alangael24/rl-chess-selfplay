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
    if sys.platform == 'darwin':
        compile_args.append('-stdlib=libc++')

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
    setup_content = f'''
from setuptools import setup, Extension
import numpy as np
setup(
    name="chess-binding",
    ext_modules=[Extension(
        "binding",
        sources=["{os.path.join(chess_ocean_dir, "binding.c")}"],
        include_dirs=["{chess_ocean_dir}", "{np.get_include()}", "{ocean_dir}"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-std=c11"],
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


# ─── Ocean-compatible chess.py ───────────────────────────────────────────────

CHESS_PY_CONTENT = '''\
"""PufferLib Ocean Chess self-play environment.

Self-play: 2 agents per game (White=even, Black=odd), same policy.
"""

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

OBS_SIZE = 4168   # 64 board + 8 metadata + 4096 action mask
NUM_ACTIONS = 4096  # 64 * 64 (from_square * 64 + to_square)


class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=128, render_mode=None, report_interval=128,
                 max_steps=256, illegal_move_penalty=-0.1,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs * 2

        super().__init__(buf=buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents, seed,
            max_steps=max_steps,
            illegal_move_penalty=illegal_move_penalty,
            num_games=num_envs,
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get("episode_length", 0) > 0:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)
'''

# ─── Config .ini ─────────────────────────────────────────────────────────────

CHESS_INI_CONTENT = '''\
[base]
package = ocean
env_name = puffer_chess
policy_name = Chess

[env]
num_envs = 2048
max_steps = 256
illegal_move_penalty = -0.1
report_interval = 128

[vec]
backend = PufferEnv
num_envs = 1

[policy]
hidden_size = 512
num_blocks = 4

[train]
total_timesteps = 1_000_000_000
learning_rate = 2.5e-4
ent_coef = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 0.5
max_grad_norm = 0.5
update_epochs = 4
minibatch_size = 16384
bptt_horizon = 16
checkpoint_interval = 500
'''

# ─── Policy class (appended to ocean/torch.py) ──────────────────────────────

CHESS_POLICY_CONTENT = '''

# === Chess Self-Play Policy ===
CHESS_BOARD_SIZE = 64
CHESS_NUM_PIECE_TYPES = 13
CHESS_NUM_ACTIONS = 4096


class ChessResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        return F.relu(self.ln2(self.fc2(F.relu(self.ln1(self.fc1(x))))) + x)


class Chess(nn.Module):
    def __init__(self, env, hidden_size=512, num_blocks=4, **kwargs):
        super().__init__()

        self.piece_embedding = nn.Embedding(CHESS_NUM_PIECE_TYPES, 32)

        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        board_feat_size = CHESS_BOARD_SIZE * 32
        self.input_fc = pufferlib.pytorch.layer_init(
            nn.Linear(board_feat_size + 64, hidden_size))
        self.input_ln = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            ChessResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, CHESS_NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

    def forward_eval(self, x, state=None):
        batch_size = x.shape[0]
        board = x[:, :CHESS_BOARD_SIZE].long()
        meta = x[:, CHESS_BOARD_SIZE:CHESS_BOARD_SIZE + 8].float()
        action_mask = x[:, CHESS_BOARD_SIZE + 8:] > 0

        board = torch.clamp(board, 0, CHESS_NUM_PIECE_TYPES - 1)
        board_emb = self.piece_embedding(board)
        board_flat = board_emb.reshape(batch_size, -1)

        meta_feat = self.meta_encoder(meta)

        combined = torch.cat([board_flat, meta_feat], dim=1)
        x = F.relu(self.input_ln(self.input_fc(combined)))

        for block in self.blocks:
            x = block(x)

        logits = self.actor(x)
        masked_logits = logits.masked_fill(~action_mask, -1e9)

        all_masked = ~action_mask.any(dim=1)
        if all_masked.any():
            masked_logits[all_masked] = logits[all_masked]

        return masked_logits, self.critic(x)

    def forward(self, x, state=None):
        return self.forward_eval(x, state)
'''


if __name__ == '__main__':
    if '--clean' in sys.argv:
        clean()
    else:
        install()
