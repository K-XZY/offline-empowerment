"""Toy check of the symbolic-empowerment idea on a deterministic grid maze (numpy only).

Ground truth: with deterministic dynamics the n-step empowerment is E_n(s) = log |{states reachable in n steps}|
(channel capacity of a noiseless channel = log of the number of distinguishable outputs).

From a limited offline random-walk dataset we compare
  teacher   : purely data-driven plug-in, log #distinct s_{t+n} observed after s_t = s (stands in for a learned
              estimator: biased low where data is thin, undefined where s was never visited)
  mlp       : an MLP regressed on the teacher (neural smoothing of the teacher)
  programs  : a few hand-written candidate programs, standing in for what an LM might propose. They are NOT
              LM outputs. Each is scored by its Spearman correlation with the teacher on visited states
              (the "teacher as feedback" signal); the best-scoring one is reported as `selected`.
All are evaluated by MAE (nats) against the ground truth, on visited and on never-visited free cells.
"""

import argparse
import json

import numpy as np

ACTIONS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

# open rooms, corridors and dead ends, so E_n(s) varies a lot across states
MAZE = [
    '#################',
    '#.......#.......#',
    '#.......#.......#',
    '#.......#.......#',
    '#...............#',
    '#.......#.......#',
    '####.#######.####',
    '#...#.#...#.#...#',
    '#.#.#.#.#.#.#.#.#',
    '#.#...#.#...#.#.#',
    '#.#####.#####.#.#',
    '#.......#.......#',
    '#.#######.#####.#',
    '#.........#.....#',
    '#################',
]


def make_env(teleport):
    free = np.array([[c == '.' for c in row] for row in MAZE])
    cells = [tuple(p) for p in np.argwhere(free)]
    idx = {c: i for i, c in enumerate(cells)}
    tele = {(1, 1): (13, 15)} if teleport else {}  # stepping onto (1,1) sends you to (13,13)
    nxt = np.zeros((len(cells), len(ACTIONS)), dtype=int)
    for i, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(ACTIONS):
            t = (r + dr, c + dc)
            t = t if free[t] else (r, c)
            t = tele.get(t, t) if a != 0 else t
            nxt[i, a] = idx[t]
    return cells, nxt


def reach_counts(nxt, n, unknown=None):
    """|reachable in n steps| per state; `unknown[s, a]` transitions are treated as 'stay' (pessimistic)."""
    S = nxt.shape[0]
    table = nxt if unknown is None else np.where(unknown, np.arange(S)[:, None], nxt)
    reach = np.eye(S, dtype=bool)
    for _ in range(n):
        new = np.zeros_like(reach)
        for a in range(table.shape[1]):
            new |= reach @ np.eye(S, dtype=bool)[table[:, a]]
        reach = new
    return reach.sum(1)


def collect(nxt, num_transitions, ep_len, start_states, rng):
    trajs = []
    total = 0
    while total < num_transitions:
        s = rng.choice(start_states)
        traj = [s]
        acts = rng.integers(0, len(ACTIONS), ep_len)
        for a in acts:
            s = nxt[s, a]
            traj.append(s)
        trajs.append((np.array(traj), acts))
        total += ep_len
    return trajs


def teacher_estimate(trajs, S, n):
    succ = [set() for _ in range(S)]
    for states, _ in trajs:
        for t in range(len(states) - n):
            succ[states[t]].add(states[t + n])
    counts = np.array([len(x) for x in succ])
    return np.where(counts > 0, np.log(np.maximum(counts, 1)), np.nan)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_mlp(X, y, Xall, rng, steps=3000, width=64, lr=3e-3):
    """Tiny 2-hidden-layer tanh MLP with Adam, full batch."""
    shapes = [(X.shape[1], width), (width,), (width, width), (width,), (width, 1), (1,)]
    params = [rng.normal(0, 1 / np.sqrt(s[0]), s) if len(s) == 2 else np.zeros(s) for s in shapes]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        W1, b1, W2, b2, W3, b3 = params
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = (h2 @ W3 + b3)[:, 0]
        g = 2 * (out - y)[:, None] / len(y)
        gW3, gb3 = h2.T @ g, g.sum(0)
        g2 = (g @ W3.T) * (1 - h2**2)
        gW2, gb2 = h1.T @ g2, g2.sum(0)
        g1 = (g2 @ W2.T) * (1 - h1**2)
        gW1, gb1 = X.T @ g1, g1.sum(0)
        for i, gr in enumerate([gW1, gb1, gW2, gb2, gW3, gb3]):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr**2
            params[i] = params[i] - lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    W1, b1, W2, b2, W3, b3 = params
    return (np.tanh(np.tanh(Xall @ W1 + b1) @ W2 + b2) @ W3 + b3)[:, 0]


def programs(cells, trajs, nxt_shape, n):
    """Candidate symbolic estimators. Each only sees the offline data (never the true transition table)."""
    S, A = nxt_shape
    coords = np.array(cells)
    seen = np.zeros(S, dtype=bool)
    learned = np.zeros((S, A), dtype=int)
    known = np.zeros((S, A), dtype=bool)
    for states, acts in trajs:
        seen[states] = True
        learned[states[:-1], acts] = states[1:]
        known[states[:-1], acts] = True

    out = {}
    out['P0_constant'] = np.full(S, np.log(n + 1))
    # P1: geometric prior -- log #free cells within Manhattan radius n (ignores connectivity and teleports)
    d = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
    out['P1_manhattan_ball'] = np.log((d <= n).sum(1))
    # P2: local-openness heuristic -- log of (1 + #open neighbours)^... , a cheap 'corridor vs junction' prior
    open_nb = np.array([sum((r + dr, c + dc) in set(cells) for dr, dc in ACTIONS[1:]) for r, c in cells])
    out['P2_local_openness'] = n * np.log1p(open_nb) / 2
    # P3: model-based program -- "the world is a deterministic grid": fit the 1-step table from data
    # (unseen (s, a) -> stay), then E_n = log |n-step reachable set| by composition.
    out['P3_det_model_bfs'] = np.log(reach_counts(learned, n, unknown=~known))
    # P4: same, but unseen (s, a) fall back to the geometric move if the target cell was ever visited
    # ("unseen moves into known-open cells probably work") -- a stronger, possibly wrong, prior
    cell_idx = {c: i for i, c in enumerate(cells)}
    guess = learned.copy()
    for i, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(ACTIONS):
            if not known[i, a]:
                j = cell_idx.get((r + dr, c + dc))
                guess[i, a] = j if (j is not None and seen[j]) else i
    out['P4_det_model_bfs_optimistic'] = np.log(reach_counts(guess, n))
    return out, seen


def run(seed, num_transitions, n, teleport, ep_len=50):
    rng = np.random.default_rng(seed)
    cells, nxt = make_env(teleport)
    S = len(cells)
    truth = np.log(reach_counts(nxt, n))
    # offline data starts in the top-left quarter only -> coverage is uneven, as in real offline datasets
    starts = [i for i, (r, c) in enumerate(cells) if r <= 7 and c <= 7]
    trajs = collect(nxt, num_transitions, ep_len, starts, rng)
    teach = teacher_estimate(trajs, S, n)
    visited = ~np.isnan(teach)
    visits = np.zeros(S)
    for states, _ in trajs:
        np.add.at(visits, states[:-n], 1)
    # only trust the teacher where it has seen many n-step windows (top quartile of visited states)
    reliable = visited & (visits >= np.quantile(visits[visited], 0.75))

    coords = np.array(cells, dtype=float) / 16.0
    mlp = fit_mlp(coords[visited], teach[visited], coords, rng)

    progs, _ = programs(cells, trajs, nxt.shape, n)
    def score(v, mask):
        return spearman(v[mask], teach[mask]) if np.std(v[mask]) > 0 else -1.0

    scores = {k: score(v, visited) for k, v in progs.items()}
    scores_rel = {k: score(v, reliable) for k, v in progs.items()}
    selected = max(scores, key=scores.get)
    selected_rel = max(scores_rel, key=scores_rel.get)

    def mae(est, mask):
        return float(np.mean(np.abs(est[mask] - truth[mask]))) if mask.any() else float('nan')

    unvisited = ~visited
    teach_filled = np.where(visited, teach, np.nanmean(teach))
    res = dict(
        seed=seed,
        N=num_transitions,
        frac_visited=float(visited.mean()),
        teacher=(mae(teach_filled, visited), mae(teach_filled, unvisited)),
        mlp=(mae(mlp, visited), mae(mlp, unvisited)),
        selected=selected,
        selected_mae=(mae(progs[selected], visited), mae(progs[selected], unvisited)),
        selected_rel=selected_rel,
        selected_rel_mae=(mae(progs[selected_rel], visited), mae(progs[selected_rel], unvisited)),
        truth_std=float(truth.std()),
        programs={
            k: dict(score=scores[k], score_rel=scores_rel[k], mae_vis=mae(v, visited), mae_unvis=mae(v, unvisited))
            for k, v in progs.items()
        },
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    all_res = []
    for teleport in (False, True):
        print(f'\n=== teleport={teleport}, n={args.n} ===')
        print('MAE vs truth in nats, shown as visited / never-visited states')
        print(f'{"N":>7} {"visited":>8} | {"teacher":>13} | {"mlp(teacher)":>13} | {"sel: all":>13} | {"sel: reliable":>13}  picks (reliable)')
        for N in (500, 2000, 10000, 50000):
            rs = [run(s, N, args.n, teleport) for s in range(args.seeds)]
            all_res += [dict(teleport=teleport, **r) for r in rs]
            f = lambda key, j: np.nanmean([r[key][j] for r in rs])
            sel = {}
            for r in rs:
                sel[r['selected_rel']] = sel.get(r['selected_rel'], 0) + 1
            print(
                f'{N:>7} {np.mean([r["frac_visited"] for r in rs]):>8.2f} | '
                f'{f("teacher", 0):>5.3f} / {f("teacher", 1):>5.3f} | {f("mlp", 0):>5.3f} / {f("mlp", 1):>5.3f} | '
                f'{f("selected_mae", 0):>5.3f} / {f("selected_mae", 1):>5.3f} | '
                f'{f("selected_rel_mae", 0):>5.3f} / {f("selected_rel_mae", 1):>5.3f}  {sel}'
            )
        print(f'  truth std over states: {all_res[-1]["truth_std"]:.3f} nats')
        print('  per-program (N=2000): Spearman with teacher on all visited / reliable states; MAE vs truth')
        for k in all_res[-1]['programs']:
            rs2 = [r for r in all_res if r['teleport'] == teleport and r['N'] == 2000]
            sc = np.mean([r['programs'][k]['score'] for r in rs2])
            sr = np.mean([r['programs'][k]['score_rel'] for r in rs2])
            mv = np.mean([r['programs'][k]['mae_vis'] for r in rs2])
            mu = np.nanmean([r['programs'][k]['mae_unvis'] for r in rs2])
            print(f'    {k:<28} score={sc:+.2f} / {sr:+.2f}  mae={mv:.3f} / {mu:.3f}')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(all_res, fh, indent=1)


if __name__ == '__main__':
    main()
