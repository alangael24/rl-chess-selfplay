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


# --- Ocean-compatible chess.py ---

CHESS_PY_CONTENT = '''\
"""PufferLib Ocean Chess self-play environment.

Self-play: 2 agents per game (White=even, Black=odd), same policy.
Two-phase action system: 97 actions (pick piece, pick dest, pass).
"""

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

OBS_SIZE = 301    # 64 board + 2 side + 4 castling + 1 ep + 2 phase + 64 selected
                  # + 64 valid_pieces + 64 valid_dests + 32 valid_promos
                  # + 1 self_check + 1 opp_check + 1 rule50 + 1 pass_valid
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
                 fen_file=None, fen_curric_pct=0.0,
                 buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs * 2

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
            fen_curric_pct=fen_curric_pct,
            num_games=num_envs,
        )
        if fen_file is not None:
            init_kwargs['fen_file'] = fen_file

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            self.num_agents, seed,
            **init_kwargs,
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

# --- Config .ini ---

CHESS_INI_CONTENT = '''\
[base]
package = ocean
env_name = puffer_chess
policy_name = Chess
rnn_name = ChessLSTM

[env]
num_envs = 2048
max_steps = 256
illegal_move_penalty = -0.1
reward_invalid_piece = -0.01
reward_invalid_move = -0.01
reward_valid_piece = 0.0
reward_valid_move = 0.0
reward_capture_bonus = 0.0
reward_check_bonus = 0.0
reward_repetition = 0.0
reward_material = 0.0
reward_position = 0.0
reward_castling = 0.0
reward_draw = 0.0
reward_see_hanging = 0.0
enable_50_move_rule = 1
enable_threefold_repetition = 1
fen_curric_pct = 0.0
report_interval = 128

[vec]
backend = PufferEnv
num_envs = 1

[policy]
hidden_size = 256
num_blocks = 2

[rnn]
input_size = 256
hidden_size = 256

[train]
total_timesteps = 1_000_000_000
precision = bfloat16
learning_rate = 1e-4
ent_coef = 0.005
gamma = 0.997
gae_lambda = 0.95
clip_coef = 0.15
vf_coef = 0.5
max_grad_norm = 1.0
update_epochs = 1
batch_size = 131072
minibatch_size = 32768
bptt_horizon = 128
checkpoint_interval = 500
anneal_lr = True
'''

# --- Policy class (appended to ocean/torch.py) ---

CHESS_POLICY_CONTENT = '''

# === Chess Self-Play Policy ===
CHESS_BOARD_SIZE = 64
CHESS_NUM_PIECE_TYPES = 13
CHESS_NUM_ACTIONS = 97

# Observation layout offsets
CHESS_OBS_BOARD = 0
CHESS_OBS_SIDE = 64
CHESS_OBS_CASTLING = 66
CHESS_OBS_EP = 70
CHESS_OBS_PHASE = 71
CHESS_OBS_SELECTED = 73
CHESS_OBS_VALID_PIECES = 137
CHESS_OBS_VALID_DESTS = 201
CHESS_OBS_VALID_PROMOS = 265
CHESS_OBS_SELF_CHECK = 297
CHESS_OBS_OPP_CHECK = 298
CHESS_OBS_RULE50 = 299
CHESS_OBS_PASS_VALID = 300


class ChessResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, c)
        self.gn2 = nn.GroupNorm(8, c)

    def forward(self, x):
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return F.relu(y + x)


class Chess(nn.Module):
    def __init__(self, env, hidden_size=256, num_blocks=2, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        conv_in = CHESS_NUM_PIECE_TYPES + 3
        conv_channels = 64
        self.board_stem = nn.Conv2d(
            conv_in, conv_channels, kernel_size=3, padding=1, bias=False)
        self.board_gn = nn.GroupNorm(8, conv_channels)
        self.board_blocks = nn.ModuleList([
            ChessResidualBlock(conv_channels) for _ in range(num_blocks)
        ])
        self.board_proj = pufferlib.pytorch.layer_init(
            nn.Linear(conv_channels * 8 * 8, hidden_size))

        self.scalar_encoder = nn.Sequential(
            nn.Linear(45, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.fusion_fc = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size + 64, hidden_size))
        self.fusion_ln = nn.LayerNorm(hidden_size)

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, CHESS_NUM_ACTIONS), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)

        self._action_mask = None

    def encode_observations(self, x, state=None):
        batch_size = x.shape[0]

        board = x[:, CHESS_OBS_BOARD:CHESS_OBS_BOARD + 64].long()
        side = x[:, CHESS_OBS_SIDE:CHESS_OBS_SIDE + 2].float() / 255.0
        castling = x[:, CHESS_OBS_CASTLING:CHESS_OBS_CASTLING + 4].float() / 255.0
        ep = x[:, CHESS_OBS_EP:CHESS_OBS_EP + 1].float() / 255.0
        phase = x[:, CHESS_OBS_PHASE:CHESS_OBS_PHASE + 2].float() / 255.0
        selected = x[:, CHESS_OBS_SELECTED:CHESS_OBS_SELECTED + 64].float() / 255.0
        valid_pieces = x[:, CHESS_OBS_VALID_PIECES:CHESS_OBS_VALID_PIECES + 64].float() / 255.0
        valid_dests = x[:, CHESS_OBS_VALID_DESTS:CHESS_OBS_VALID_DESTS + 64].float() / 255.0
        valid_promos = x[:, CHESS_OBS_VALID_PROMOS:CHESS_OBS_VALID_PROMOS + 32].float() / 255.0
        self_check = x[:, CHESS_OBS_SELF_CHECK:CHESS_OBS_SELF_CHECK + 1].float() / 255.0
        opp_check = x[:, CHESS_OBS_OPP_CHECK:CHESS_OBS_OPP_CHECK + 1].float() / 255.0
        rule50 = x[:, CHESS_OBS_RULE50:CHESS_OBS_RULE50 + 1].float() / 255.0
        pass_valid = x[:, CHESS_OBS_PASS_VALID:CHESS_OBS_PASS_VALID + 1].float() / 255.0

        board = torch.clamp(board, 0, CHESS_NUM_PIECE_TYPES - 1)
        board_oh = F.one_hot(board, num_classes=CHESS_NUM_PIECE_TYPES).float()
        board_oh = board_oh.view(batch_size, 8, 8, CHESS_NUM_PIECE_TYPES).permute(0, 3, 1, 2)

        vp_plane = valid_pieces.view(batch_size, 1, 8, 8)
        vd_plane = valid_dests.view(batch_size, 1, 8, 8)
        sp_plane = selected.view(batch_size, 1, 8, 8)

        spatial = torch.cat([board_oh, vp_plane, vd_plane, sp_plane], dim=1)

        board_x = F.relu(self.board_gn(self.board_stem(spatial)))
        for block in self.board_blocks:
            board_x = block(board_x)

        board_feat = board_x.reshape(batch_size, -1)
        board_feat = F.relu(self.board_proj(board_feat))

        scalars = torch.cat([side, castling, ep, phase, self_check, opp_check, rule50, pass_valid, valid_promos], dim=1)
        scalar_feat = self.scalar_encoder(scalars)

        combined = torch.cat([board_feat, scalar_feat], dim=1)
        hidden = F.relu(self.fusion_ln(self.fusion_fc(combined)))

        # Store action mask for decode_actions
        is_phase0 = phase[:, 0:1] > 0.5
        mask = torch.zeros(batch_size, CHESS_NUM_ACTIONS, device=x.device, dtype=torch.bool)
        vp_bool = valid_pieces > 0.5
        vd_bool = valid_dests > 0.5
        mask[:, :64] = torch.where(is_phase0, vp_bool, vd_bool)
        promo_bool = valid_promos > 0.5
        mask[:, 64:96] = torch.where(is_phase0, torch.zeros_like(promo_bool), promo_bool)
        pass_bool = pass_valid.squeeze(-1) > 0.5
        mask[:, 96] = pass_bool
        self._action_mask = mask

        return hidden

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        value = self.critic(hidden)

        if self._action_mask is not None:
            mask = self._action_mask
            if mask.shape[0] != logits.shape[0]:
                mask = None
            else:
                masked_logits = logits.masked_fill(~mask, -1e9)
                all_masked = ~mask.any(dim=1)
                if all_masked.any():
                    masked_logits[all_masked] = logits[all_masked]
                logits = masked_logits

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
