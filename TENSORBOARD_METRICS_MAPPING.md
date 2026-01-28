# TensorBoard Metrics Mapping for OnPolicyRunner

## Overview
When training with `scripts/rsl_rl/train.py`, the `OnPolicyRunner` class from the rsl-rl library logs metrics to TensorBoard through a `Logger` class. This document maps the TensorBoard graphs and metrics that appear during training.

---

## Data Flow: From Environment to TensorBoard

### 1. Environment Step (Runner.learn() → env.step())
- **Location**: `OnPolicyRunner.learn()` at line 84-101 in `rsl_rl/runners/on_policy_runner.py`
- Actions are sampled from the policy
- Environment returns: `obs, rewards, dones, extras`
- The `extras` dict contains environment-specific metrics

### 2. Logging Environment Step (Logger.process_env_step())
- **Location**: `Logger.process_env_step()` at line 77-103 in `rsl_rl/utils/logger.py`
- Processes environment step data for logging
- Accumulates rewards per environment
- Tracks episode length
- Stores episode extras (from `extras["episode"]` or `extras["log"]`)

### 3. Algorithm Update
- **Location**: `OnPolicyRunner.learn()` at line 102-127 in `rsl_rl/runners/on_policy_runner.py`
- PPO algorithm computes losses: `loss_dict = self.alg.update()`
- Returns dict with keys: `value`, `surrogate`, `entropy`, and optionally `rnd`, `symmetry`

### 4. Logging Metrics (Logger.log())
- **Location**: `Logger.log()` at line 120-229 in `rsl_rl/utils/logger.py`
- Called after each training iteration
- Logs all metrics to TensorBoard via `self.writer.add_scalar(tag, value, global_step)`

---

## Complete TensorBoard Metrics Mapping

### A. LOSS METRICS (Loss/*)
**Source**: From `PPO.update()` return value (loss_dict)

```
Loss/value              → Mean value function loss (critic loss)
Loss/surrogate          → Mean surrogate (policy) loss (PPO clipped loss)
Loss/entropy            → Mean entropy bonus (policy entropy)
Loss/rnd                → Random Network Distillation loss (if enabled)
Loss/symmetry           → Symmetry regularization loss (if enabled)
Loss/learning_rate      → Current learning rate of the optimizer
```

**Location in code**: `Logger.log()` lines 147-151
```python
for key, value in loss_dict.items():
    self.writer.add_scalar(f"Loss/{key}", value, it)
self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
```

---

### B. TRAINING METRICS (Train/*)
**Source**: From `Logger.rewbuffer` and `Logger.lenbuffer` (episode statistics)

```
Train/mean_reward                  → Mean episode reward (averaged over last 100 episodes)
Train/mean_episode_length          → Mean episode length (averaged over last 100 episodes)
Train/mean_reward/time             → Mean episode reward keyed by wall-clock time (not wandb)
Train/mean_episode_length/time     → Mean episode length keyed by wall-clock time (not wandb)
```

**Location in code**: `Logger.log()` lines 170-181
```python
if len(self.rewbuffer) > 0:
    if self.cfg["algorithm"]["rnd_cfg"]:
        self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
        self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
        self.writer.add_scalar("Rnd/weight", rnd_weight, it)
    self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
    self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
```

---

### C. RND METRICS (Rnd/*)
**Source**: From RND (Random Network Distillation) module - only if `rnd_cfg` is enabled

```
Rnd/mean_extrinsic_reward    → Mean extrinsic (environment) reward
Rnd/mean_intrinsic_reward    → Mean intrinsic (curiosity) reward from RND
Rnd/weight                   → Weight/coefficient of RND loss
```

**Conditions**: Only logged if `self.cfg["algorithm"]["rnd_cfg"]` is not None

**Location in code**: `Logger.log()` lines 165-167

---

### D. PERFORMANCE METRICS (Perf/*)
**Source**: Timing information from training loop

```
Perf/total_fps          → Training throughput (steps per second)
Perf/collection_time    → Time to collect rollout data
Perf/learning_time      → Time to perform policy update
```

**Location in code**: `Logger.log()` lines 156-159
```python
fps = int(collection_size / (collect_time + learn_time))
self.writer.add_scalar("Perf/total_fps", fps, it)
self.writer.add_scalar("Perf/collection_time", collect_time, it)
self.writer.add_scalar("Perf/learning_time", learn_time, it)
```

---

### E. POLICY METRICS (Policy/*)
**Source**: From policy network (action standard deviation)

```
Policy/mean_noise_std   → Mean action noise standard deviation (exploration noise)
```

**Location in code**: `Logger.log()` line 154
```python
self.writer.add_scalar("Policy/mean_noise_std", action_std.mean().item(), it)
```

---

### F. EPISODE METRICS (Episode/*)
**Source**: From environment extras (`extras["episode"]` or `extras["log"]`)

```
Episode/<custom_key>    → Any custom metrics from the environment that have "/" in the key
```

**Explanation**: 
- The environment can return custom metrics in the `extras` dict
- Keys with "/" in them are logged directly to TensorBoard with the namespace preserved
- Keys without "/" are aggregated and logged under "Episode/"

**Location in code**: `Logger.log()` lines 130-151
```python
if self.ep_extras:
    for key in self.ep_extras[0]:
        # ... aggregate values across episodes ...
        if "/" in key:
            self.writer.add_scalar(key, value, it)
        else:
            self.writer.add_scalar("Episode/" + key, value, it)
```

---

## How Reward Terms Are Logged

### 1. Episode Reward Accumulation
```python
# From Logger.process_env_step() - lines 77-103
self.cur_reward_sum += rewards          # Accumulate rewards each step
self.cur_episode_length += 1            # Track episode length

# When episode ends (dones > 0):
self.rewbuffer.extend(self.cur_reward_sum[new_ids])  # Store final episode reward
self.lenbuffer.extend(self.cur_episode_length[new_ids])  # Store episode length
```

### 2. RND Reward Splitting (if enabled)
```python
# From Logger.process_env_step() - lines 93-101
if intrinsic_rewards is not None:
    self.cur_ereward_sum += rewards              # Extrinsic reward accumulation
    self.cur_ireward_sum += intrinsic_rewards    # Intrinsic reward accumulation
    self.cur_reward_sum += rewards + intrinsic_rewards  # Total reward
```

### 3. Reward Dictionary from Environment
The environment returns `rewards` as a tensor, but can also include custom reward breakdowns in `extras`:
- `extras["episode"]` or `extras["log"]` can contain reward components
- These are processed and logged under `Episode/<component_name>`

---

## Training Loop Call Sequence

```
OnPolicyRunner.learn() [line 62-127]
  ├── for it in range(num_iterations):
  │   ├── Rollout phase [line 84-101]:
  │   │   ├── for each env_step:
  │   │   │   ├── actions = alg.act(obs)
  │   │   │   ├── obs, rewards, dones, extras = env.step(actions)
  │   │   │   ├── alg.process_env_step(obs, rewards, dones, extras)
  │   │   │   └── logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
  │   │   └── collect_time = elapsed
  │   │
  │   ├── Learning phase [line 102-108]:
  │   │   ├── alg.compute_returns(obs)
  │   │   ├── loss_dict = alg.update()  # Returns dict with loss values
  │   │   └── learn_time = elapsed
  │   │
  │   └── Logging phase [line 110-120]:
  │       └── logger.log(it, loss_dict, collect_time, learn_time, ...)
  │           └── Calls writer.add_scalar() for all metrics
  │
  └── Save model checkpoints
```

---

## Environment Extras Dictionary Structure

The environment's `extras` dict typically contains:

```python
extras = {
    "episode": {
        "reward": float,           # Episode total reward
        "length": int,             # Episode length
        # Custom reward terms (if environment provides them):
        "reward_forward": float,   # Example: forward motion reward
        "reward_turn": float,      # Example: turning reward
        ...
    },
    "time_outs": torch.Tensor,    # Boolean indicating timeout
    # ... other environment-specific data
}
```

Custom metrics with "/" in the key will be logged directly. Others go under "Episode/".

---

## Complete Metric Summary by TensorBoard Section

| TensorBoard Group | Metric Name | Source | Type |
|---|---|---|---|
| **Loss** | value | PPO algorithm | Critic loss |
| | surrogate | PPO algorithm | Policy loss |
| | entropy | PPO algorithm | Entropy regularization |
| | rnd | RND module (optional) | Curiosity exploration loss |
| | symmetry | Symmetry module (optional) | Symmetry regularization loss |
| | learning_rate | Optimizer | Learning rate schedule |
| **Train** | mean_reward | Environment episodes | Episode return |
| | mean_episode_length | Environment episodes | Episode duration |
| | mean_reward/time | Environment episodes (non-wandb) | Return vs wall-clock time |
| | mean_episode_length/time | Environment episodes (non-wandb) | Length vs wall-clock time |
| **Rnd** | mean_extrinsic_reward | Environment + RND | Extrinsic reward (if RND enabled) |
| | mean_intrinsic_reward | RND module | Intrinsic reward (if RND enabled) |
| | weight | RND module | RND loss weight (if RND enabled) |
| **Perf** | total_fps | Timing | Steps per second |
| | collection_time | Timing | Rollout collection time |
| | learning_time | Timing | Policy update time |
| **Policy** | mean_noise_std | Policy network | Action standard deviation |
| **Episode** | \<custom_key\> | Environment extras | Custom environment metrics |

---

## Key Implementation Details

### 1. Buffers and History
- `rewbuffer`: Deque with max length 100, stores last 100 episode rewards
- `lenbuffer`: Deque with max length 100, stores last 100 episode lengths
- `ep_extras`: List accumulating episode info dicts during current iteration

### 2. Logging Frequency
- All metrics logged once per training iteration
- Iteration = one rollout collection + one policy update

### 3. Multi-GPU Handling
- Only rank 0 GPU logs metrics (` disable_logs = is_distributed and gpu_global_rank != 0`)
- Prevents duplicate logging in distributed training

### 4. Writer Implementation
Three logging backends supported:
- **TensorBoard**: PyTorch `SummaryWriter` (default)
- **Weights & Biases**: `WandbSummaryWriter` wrapper
- **Neptune**: `NeptuneSummaryWriter` wrapper

All use `.add_scalar(tag, value, global_step)` interface.

---

## File References

- **OnPolicyRunner**: `rsl_rl/runners/on_policy_runner.py` (lines 30-194)
  - Main training loop in `learn()` method
- **Logger**: `rsl_rl/utils/logger.py` (lines 21-290)
  - Environment step processing: `process_env_step()` (lines 77-103)
  - Metric logging: `log()` (lines 120-229)
- **PPO Algorithm**: `rsl_rl/algorithms/ppo.py` (lines 192-430)
  - Loss computation in `update()` method
- **Training Script**: `scripts/rsl_rl/train.py` (lines 95-215)
  - Creates runner and calls `runner.learn()`
