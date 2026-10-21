# Symbolic empowerment: notes and a toy check

Written overnight, 2026-10-21. Section 1 summarises this repo. Section 2 works through the idea. Section 3 is a small
numpy experiment (`toy_gridworld.py`, output in `results.txt`). Sections 4–6 cover what the idea might enable, where
it could apply in robotics, and a concrete plan. Claims about other papers come from web searches (linked at the end).
Anything about what symbolic empowerment *would* do is a hypothesis unless the toy experiment shows it, and the toy
experiment is small.

---

## 1. Digest of `offline-empowerment`

**What it's for.** It estimates empowerment from **unlabeled offline data**: no rewards and no environment
interaction. That estimate then serves as an **exploration bonus** when an agent later learns a goal-reaching task
online.

**What empowerment means here** (`agents/empowerment_skill.py`). There are K discrete skills z (default K = 15) and a
skill-conditioned policy π(a | s, z). The quantity is

  Ê(s) = I(s⁺ ; z | s),  with z ~ Unif(K) and s⁺ a future state drawn from the policy's discounted occupancy.

In words: how distinguishable the futures of different skills are from state s. Because z is uniform over K values,
**Ê(s) ≤ log K ≈ 2.71 nats**. This is the *effective* empowerment of the skill set the method can learn from the data,
not the physical channel capacity of the robot.

**How it is trained** (offline, `train_offline.py`; defaults from `get_config`):
- **Occupancy models.** Q^z(s⁺ | s, a) and V^z(s⁺ | s) are learned by TD on dataset transitions with a clipped
  "linexp" loss. Each is parameterised as a Gaussian in a 256-d latent: log Q = −‖φ(s,a,z) − ψ(s⁺)‖² / d, with ψ
  shared. A "self term" handles s⁺ = s. Random dataset states serve as s⁺.
- **Skill policy.** The policy π(a | s, z) is trained to maximise the effective empowerment E_Δ (eqs. 11–13), with the
  gradient flowing only through Q^z of its own action. Actions get fixed Gaussian noise σ_a (default 0.1). An optional
  BC term (α_bc, default 0) keeps the policy near the data.
- **Networks and schedule.** Q, V and π are 3×512 MLPs with LayerNorm. Defaults: Adam 3e-4, batch 1024, 1M steps,
  target nets with τ = 0.005, γ = 0.99.
- **Data.** OGBench datasets (antmaze / pointmaze-teleport / antsoccer / cube-single). The README doesn't state compute
  cost, and I didn't run it.
- **Reading out Ê(s).** A Monte Carlo estimate that loops over all K skills and draws N s⁺ samples from V^z in latent
  space (`empowerment()`, N = 64 online).

**How it is used online** (`agents/online_crl.py`, `utils/rlpd.py`, `utils/rollout.py`):
- The frozen estimator labels every replay row (offline and online).
- The distillation target for every row of a trajectory is **the max of Ê over that trajectory**.
- A twin-head B(s, a) regresses that target minus a constant E_mean.
- The CRL actor maximises Q(s, a, g) + β · min B(s, a). RLPD mixes in the offline data. RND can be added on top. SUPE
  (skills plus online high-level policy) is the second base learner.

**What it enables.** A reward-free, dataset-only signal for which states give the agent the most control. The README
reports it helps exploration in sparse goal-reaching tasks, and it is tested against four offline-empowerment
baselines (CF, CF-FlowBC, DV, DADS).

**What it takes, in one line.** A diverse enough offline dataset, about 1M gradient steps on five networks with
double-nested sampling, then an extra distillation network online.

**One detail that matters for your idea.** Because the bonus target is a *trajectory max*, any spurious spike of Ê
anywhere in a trajectory labels the whole trajectory as high-bonus. Non-smooth or outlier-prone Ê is therefore
amplified, not averaged out, by the current design. This follows from reading the code, not from an experiment.
Possible sources of non-smoothness (not tested):
- MC noise (64 samples × 15 skills)
- TD bootstrapping error
- unconstrained extrapolation of φ/ψ off the data manifold
- saturation near the log K ceiling

---

## 2. The idea, made precise

"Symbolic empowerment" = represent Ê, or part of its pipeline, as **a program an LM writes**, with a few fitted
constants, rather than as a neural net. There are three levels, from shallowest to deepest:

| Level | What the LM writes | How Ê is computed | What scores the program |
|---|---|---|---|
| **L1 – direct** | `E(obs) -> float` over named state features (xy, joint angles, distance-to-wall, cube pose…) | evaluate the program | agreement with the neural Ê on well-covered states (distillation) |
| **L2 – structural** | features / abstraction ψ(s⁺) the occupancy is defined over, or the occupancy's functional form | the paper's own TD estimator on top | the estimator's own held-out TD / occupancy losses, so **no teacher is needed** |
| **L3 – model-based** | a simplified code world model (e.g. "deterministic grid / point-mass with walls"), fitted to the offline transitions | channel capacity of the model: log reachable-set size if deterministic, Blahut–Arimoto or a linear-Gaussian approximation (Salge et al.) if not | one-step prediction error on held-out data, plus agreement with the neural Ê |

The loop you describe ("their way of learning empowerment becomes a signal for improving the LM") looks like
FunSearch / LLM-SR / Eureka. The LM proposes programs, an automatic scorer ranks them, and the best ones go back into
the prompt. Optionally, that score can serve as a reward to fine-tune the LM itself. Eureka does this for reward code
and LLM-SR for equation skeletons with fitted constants. GIF-MCTS does it for code world models checked against
trajectories, which is close to L3. In my searches I didn't find work that does this for empowerment, but I can't
rule out that it exists.

**Candidate scoring signals,** cheapest first:
1. Rank / value agreement with the neural Ê, **restricted to states where Ê is trustworthy** (high data density,
   low MC variance).
2. Hard sanity constraints: 0 ≤ Ê ≤ log K (or ≤ n log|A| for n-step), and invariance to features that shouldn't
   matter (e.g. goal position).
3. Held-out TD loss of the estimator when the program supplies its structure (L2), or held-out one-step error of the
   code model (L3).
4. Downstream: online exploration success when the program replaces the neural bonus. This is the real objective but
   costs a full online run per candidate, so use it only for the final few.

**Two directions of coupling,** both worth having:
- **Neural → symbolic (your proposal):** the neural Ê scores LM programs.
- **Symbolic → neural:** use the program as a prior. Either Ê = E_sym(s) + NN residual (small and regularised), or
  E_sym as an extra regression target or regulariser off-distribution. The neural part is then only asked to correct
  the program where data exists, and the program supplies smooth extrapolation elsewhere.

**One conceptual catch to decide early.** The paper's Ê is the empowerment of skills the *data and the method* can
represent, capped at log K. A physics-style program (L3) computes something closer to *true* channel capacity. The two
differ: a large open region that the dataset barely covers has high true empowerment but low data-supported
empowerment. For exploration you probably want the former. For safe offline control you may want the latter. Scoring
an L3 program against the neural Ê implicitly asks for the latter.

---

## 3. Toy check (`toy_gridworld.py`)

**Setup.** A 15×17 maze with rooms, corridors and dead ends; 5 actions; deterministic, plus a variant with a one-way
teleporter. The ground truth is exact: E₄(s) = log |states reachable in 4 steps|. The offline data are random walks
that start only in the top-left quarter, so coverage is uneven. Compared estimators:
- **Teacher:** a purely data-driven plug-in, log #distinct s_{t+4} observed after s. It stands in for a learned
  estimator and is biased low where data is thin. **It is not the paper's estimator.**
- **MLP regressed on the teacher:** "just smooth the teacher".
- **Five hand-written programs, my stand-ins for LM proposals.** **No LM was called.**
  - P0: constant
  - P1: log #free cells in a Manhattan ball
  - P2: a local-openness heuristic
  - P3: an L3 program that fits a deterministic one-step table from data (unseen (s, a) → stay), then takes
    log |reachable set| by composition
  - P4: like P3, but unseen moves into known-open cells are assumed to succeed

Each program is scored by Spearman correlation with the teacher, on all visited states and on the "reliable" top
quartile by visit count. The top-scoring program is reported. Everything is evaluated by MAE (nats) against the truth.
5 seeds.

**Results** (no teleporter; the teleporter variant looks the same, see `results.txt`):

| offline transitions | frac. states visited | teacher MAE (visited / unvisited) | MLP(teacher) | selected program |
|---|---|---|---|---|
| 500    | 0.41 | 1.65 / 1.12 | 1.65 / 2.47 | 0.41 / 2.48 |
| 2 000  | 0.53 | 1.08 / 0.59 | 1.08 / 3.25 | 0.25 / 2.40 |
| 10 000 | 0.68 | 0.74 / 0.35 | 0.74 / 2.55 | 0.14 / 2.32 |
| 50 000 | 0.79 | 0.43 / 0.23 | 0.44 / 1.81 | 0.03 / 2.26 |

(The std of the truth across states is about 0.5 nats.)

What the toy shows, and only in this toy:
1. **Structure turns n-step coverage into 1-step coverage.** The model-based program (P3) only needs each (s, a) seen
   once and composes them. The data-driven teacher needs the n-step action *sequences* to occur in the data. On visited
   states the selected program is 4–15× more accurate than the teacher at every data size. The 50k-transition program
   (0.03) is far better than the teacher at 50k (0.43).
2. **Smoothing the teacher with a neural net doesn't fix it.** The MLP inherits the teacher's coverage bias. The gain
   comes from the *hypothesis class*, not from smoothness as such.
3. **The teacher-as-scorer leaks its own bias into selection.** At 2k transitions, P4 is most accurate on visited
   states (0.13 vs 0.25), but P3 scores higher (ρ 0.81 vs 0.68). Both the teacher and P3 underestimate where coverage
   is thin, so they agree with each other. Restricting the score to reliable states partly helps (ties at 0.77). This
   is the main methodological warning for the real version: **an imperfect neural Ê will select programs that share
   its errors.**
4. **Nothing checks behaviour off the data.** On never-visited states the model-based programs are badly wrong (about
   2.3 nats; unseen ⇒ "stay"). The simple geometric priors (P1, P2) extrapolate much better (0.15–0.5), but the
   teacher score can't see that, because there's no teacher there. Generalisation beyond coverage has to come from the
   prior itself, or from a scorer that doesn't depend on the data (constraints, downstream task).

Toy limitations:
- tabular, deterministic, and no learned features
- the "LM" was me writing five programs
- the teacher is a counting estimator, not the paper's TD estimator
- the teleporter barely changes the numbers, because it affects few states (the variant doesn't discriminate between
  priors)

Run it with `python research/symbolic_empowerment/toy_gridworld.py --out results.json` (numpy only, about 1 min on
CPU).

---

## 4. What it might enable (hypotheses, ordered by how plausible I find them)

- **Sample efficiency of the estimator.** Point 1 of the toy is the mechanism. When the LM's prior about dynamics is
  right, far fewer transitions are needed. The obvious real test: subsample the OGBench datasets (1%, 10%, 100%) and
  compare against the neural Ê.
- **Cheap, smooth, bounded bonuses online.** The online loop needs `empowerment_np` (15 skills × 64 samples per state)
  plus a distillation network. A closed-form E_sym(obs) is near-free to evaluate, can be bounded by construction, and
  removes the trajectory-max spike problem. Whether it *helps* exploration is untested: a smoother bonus could also be
  a less informative one.
- **Interpretability and debugging.** You can read *why* a state scores high ("distance to nearest wall in xy";
  "cube not at table edge and within gripper reach") and see when it's wrong.
- **Transfer across layouts.** A program parameterised by a map or object list may move to a new maze or table layout
  without retraining. The neural Ê is tied to one dataset.
- **Better extrapolation off-distribution,** *only* if the prior is right (toy point 4). This is not automatic.
- **An LM-training signal,** as you suggest: (program, score) pairs from the estimator are a verifiable reward for
  improving an LM's physical/agentic reasoning. That's a research direction in itself, and the evaluator-bias issue
  (toy point 3) applies directly.

Main risks:
- The LM prior is wrong for contact-rich dynamics (the ant's own gait, cube pushing).
- The score gets gamed by agreeing with the teacher's errors.
- Pixel observations need a perception step before any symbolic feature exists. `envs/maze.py` has a pixel variant.
- Symbolic E mostly captures *geometry*, while the paper's Ê also reflects *what the data's behaviour supports* (§2
  catch).

---

## 5. Where in robotics it could apply (candidates, untested)

- **Navigation** (antmaze, pointmaze, legged or wheeled robots in buildings). Empowerment is largely a function of xy
  and local free space, which an LM can express easily. This is the best first target and matches the repo's
  antmaze / pointmaze-teleport tasks.
- **Object-centric manipulation** (cube-single, rearrangement). Empowerment over object pose roughly means "can I
  still move the object to many places": the object is reachable, not wedged in a corner or at the table edge, and the
  gripper is free. LMs know these affordances. The robot's own arm dynamics can stay neural (L2 hybrid).
- **Ball / tool control** (antsoccer). Control over the ball depends on the agent–ball relative position and the ball's
  distance to walls, which has a compact symbolic form.
- **Safety and reversibility.** Low empowerment flags stuck or irreversible states: cliffs, tipping, jammed objects.
  Symbolic forms give auditable "keep options open" constraints, which matters more on hardware than in simulation.
- **Assistance.** Maximising a *human's* empowerment as an assistive objective is an existing line of work (Myers et
  al., "Learning to Assist Humans without Inferring Rewards", in the search results). A symbolic model of the human's
  options could be easier to trust than a neural one.

---

## 6. A concrete plan in this repo

1. **Look at the neural Ê first.** For a trained antmaze / pointmaze-teleport estimator, dump Ê at all dataset states,
   bin by xy, and measure non-smoothness (variance within bin, MC variance across seeds). This tests your "not smooth"
   observation and tells you which states are trustworthy as a scorer (toy point 3).
2. **L1 loop on navigation.** Prompt an LM with the env description, the observation layout, and the `Ê ≤ log K`
   bound. Have it write `E_sym(obs, params)`, fit params to Ê on reliable states, and score on held-out reliable states
   plus constraints. Run a FunSearch-style loop over a few dozen to hundreds of candidates.
3. **Swap it into the online loop.** Replace `agent.empowerment_np` (used in `utils/rlpd.py` and `utils/rollout.py`)
   with E_sym, keeping everything else fixed. Compare against the neural bonus, RND and no bonus on the four online
   tasks. Also try per-step E_sym instead of the trajectory max.
4. **Data-efficiency sweep.** Subsample the offline dataset and compare neural Ê, E_sym, and the residual hybrid
   E_sym + NN.
5. **L2 / L3** if 2–4 look promising: LM-written ψ features or a code world model, scored by held-out TD or one-step
   loss. These scorers don't share the teacher's coverage bias.
6. **LM improvement**, only after the scorer is shown to be unbiased enough: use the step-2 scores as rewards for
   preference or RL fine-tuning of the program-writer.

---

### Sources (from web searches, 2026-10-21)
- Eureka: LLM evolutionary reward-code search — https://arxiv.org/abs/2310.12931
- FunSearch (program search with an automatic evaluator) — https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/
- LLM-SR (LLM equation skeletons + fitted constants) — https://arxiv.org/abs/2404.18400
- Code world models via GIF-MCTS — https://arxiv.org/abs/2405.15383
- LLM-written progress functions + count bonuses for exploration — https://arxiv.org/pdf/2410.09187
- PROF (LLM reward code selected with offline data) — https://arxiv.org/pdf/2511.13765
- Salge, Glackin, Polani, empowerment as linear Gaussian channel — https://uhra.herts.ac.uk/handle/2299/9659
- Estimating the empowerment of LM agents (EELMA) — https://arxiv.org/abs/2509.22504
- Learning to Assist Humans without Inferring Rewards — https://arxiv.org/pdf/2411.02623
- NeurIPS 2026 listing "Offline Empowerment Enables Diverse Skill Discovery" (possibly related to this repo's paper; not verified) — https://neurips.cc/virtual/2026/162003
