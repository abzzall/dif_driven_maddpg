"""Read-only instrumentation of frozen-policy endpoint collision projection."""
import argparse
import os
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from eval_frozen import build_from_meta, run_episode


def instrument(env):
    original = env._update_positions
    env.audit = []

    def measured():
        active = ~env.get_dones_tensor()
        proposed = env.agent_pos.clone()
        theta = env.agent_dir[active]
        proposed[active] += torch.stack((env.agent_vel_lin[active] * torch.cos(theta),
                                        env.agent_vel_lin[active] * torch.sin(theta)), dim=1)
        pairs = torch.triu(torch.ones((env.num_agents, env.num_agents),
                                      device=env.device, dtype=torch.bool), diagonal=1)
        pairs &= active[:, None] | active[None, :]
        threshold = 2 * env.agent_radius - 1e-6
        before = (torch.cdist(proposed, proposed) < threshold) & pairs
        original()
        displacement = (env.agent_pos - proposed).norm(dim=1)
        affected = (displacement > 1e-6) & active
        after = (torch.cdist(env.agent_pos, env.agent_pos) < threshold) & pairs
        env.audit.append(dict(proposed_pair_events=int(before.sum()),
                              proposed_overlap_steps=int(before.any()),
                              intervention_steps=int(affected.any()),
                              affected_robot_steps=int(affected.sum()),
                              active_robot_steps=int(active.sum()),
                              correction_distance=float(displacement[active].sum()),
                              max_correction_distance=float(displacement.max()),
                              residual_pair_events=int(after.sum())))
    env._update_positions = measured


def fingerprint(run):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.iterdir()) if p.name == 'meta.json' or p.suffix == '.pth'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for locked evaluation')
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        for arm in ('C', 'B', 'B_wide'):
            run = repo / f'review_round1_runs/r2_3_additional_seeds/runs/{arm}_n5_seed6'
            env, algo, _ = build_from_meta(run)
            plain = run_episode(env, algo, 900000, 500, False)
            state = env.agent_pos.clone()
            instrument(env)
            measured = run_episode(env, algo, 900000, 500, False)
            assert plain == measured and torch.equal(state, env.agent_pos), (plain, measured)
        # Synthetic overlap, free movement, and completed-body cases.
        for overlap, completed in [(False, False), (True, False), (True, True)]:
            env.reset_tensor(seed=900001)
            env.agent_pos[:] = torch.arange(5, device=env.device)[:, None] * 10
            if overlap:
                env.agent_pos[1] = env.agent_pos[0] + torch.tensor([float(env.agent_radius), 0.], device=env.device)
            env.agent_vel_lin.zero_()
            env.agent_vel_ang.zero_()
            env.dones.zero_()
            if completed:
                env.dones[1] = True
            env._update_positions()
            assert bool(env.audit[-1]['proposed_pair_events']) == overlap
            assert bool(env.audit[-1]['intervention_steps']) == overlap
        print('PASS: unchanged full episode and synthetic overlap/free/completed-body cases', flush=True)
        return
    while True:
        missing = []
        for seed in range(1, 11):
            for arm in ('C', 'B', 'B_wide'):
                name = f'{arm}_n5_seed{seed}'
                target = args.out / f'{name}.csv'
                if target.exists():
                    with target.open() as f:
                        rows = list(csv.DictReader(f))
                    assert len(rows) == 100 and [int(r['scenario_seed']) for r in rows] == list(range(900000, 900100))
                    continue
                root = Path(os.environ.get("PPO_STUDY_ROOT", Path.home() / "dif_driven_ppo_study")) / "runs" if seed <= 5 else repo / 'review_round1_runs/r2_3_additional_seeds/runs'
                run = root / name
                if not (run / 'meta.json').exists():
                    missing.append(str(run))
                    continue
                env, algo, meta = build_from_meta(run)
                assert meta['reward_design'] == 'new' and int(meta['n']) == 5
                assert meta['seed'] == seed and meta.get('finished') is True
                assert meta['observation'] == ('graph' if arm == 'C' else 'flat')
                assert meta.get('mlp_hidden') == (512 if arm == 'B_wide' else None)
                assert env.num_obstacles == 0
                provenance = fingerprint(run)
                instrument(env)
                rows = []
                for scenario in range(900000, 900100):
                    env.audit.clear()
                    outcome = run_episode(env, algo, scenario, 500, False)
                    row = dict(run=name, arm=arm, training_seed=seed, scenario_seed=scenario, **outcome)
                    for key in env.audit[0]:
                        values = [s[key] for s in env.audit]
                        row[key] = max(values) if key.startswith('max_') else sum(values)
                    row['episode_intervened'] = int(row['intervention_steps'] > 0)
                    rows.append(row)
                assert fingerprint(run) == provenance
                (args.out / f'{name}.provenance.json').write_text(json.dumps(provenance, indent=2))
                temporary = target.with_suffix('.csv.partial')
                with temporary.open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(target)
                (args.out / 'status.json').write_text(json.dumps(dict(state='running', last_completed=name,
                    completed=len(list(args.out.glob('*.csv'))), updated=time.time()), indent=2))
                print(f'FINISHED {name}: 100 episodes', flush=True)
                del env, algo
                torch.cuda.empty_cache()
        status = dict(state='waiting_for_checkpoints' if missing else 'complete', missing=missing,
                      completed=len(list(args.out.glob('*.csv'))), updated=time.time())
        (args.out / 'status.json').write_text(json.dumps(status, indent=2))
        if not missing:
            import pandas as pd
            data = pd.concat([pd.read_csv(p) for p in sorted(args.out.glob('*_n5_seed*.csv'))], ignore_index=True)
            summaries = []
            for (arm, seed), group in data.groupby(['arm', 'training_seed']):
                for subset, part in [('all', group), ('successful', group[group.success == 1]),
                                     ('unsuccessful', group[group.success == 0])]:
                    if part.empty:
                        continue
                    summaries.append(dict(arm=arm, seed=int(seed), subset=subset, episodes=len(part),
                        success_pct=100 * part.success.mean(), episode_intervention_pct=100 * part.episode_intervened.mean(),
                        intervention_step_pct=100 * part.intervention_steps.sum() / part.steps.sum(),
                        affected_active_robot_step_pct=100 * part.affected_robot_steps.sum() / part.active_robot_steps.sum(),
                        correction_distance_per_episode=part.correction_distance.mean(),
                        maximum_correction_distance=part.max_correction_distance.max(),
                        proposed_pairs_per_episode=part.proposed_pair_events.mean(),
                        residual_pairs_per_episode=part.residual_pair_events.mean()))
            (args.out / 'seed_summary.json').write_text(json.dumps(summaries, indent=2))
            print('COMPLETE: all 30 policies, 3000 episodes', flush=True)
            return
        print(f'WAITING: {len(missing)} missing checkpoints; check PPO_STUDY_ROOT. Retrying in 60 s.', flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
