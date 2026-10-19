"""Fail-closed evidence validation and paired-query descriptive statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np


class EvidenceError(ValueError):
    """An archived result cannot support the requested comparison."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'evidence/SHA256SUMS.json').read_text())
    require(isinstance(manifest, dict) and bool(manifest), 'Empty or invalid manifest')
    for name, entry in manifest.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root), f'Manifest path escapes root: {name}')
        require(path.is_file(), f'Missing archive file: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'],
                f'SHA256 mismatch: {name}')
    return len(manifest)


def rows_by_query(data, expected=64):
    require(data.get('complete') is True, 'Incomplete run')
    require(not data.get('failures'), 'Run contains correctness failures')
    records = data['records']
    require(len(records) == expected, f'Expected {expected} query records')
    rows = {}
    for row in records:
        qid = row['query_id']
        require(isinstance(qid, str) and qid not in rows, f'Duplicate/invalid query ID: {qid}')
        rows[qid] = row
    return rows


def median_time(values, repeats=5):
    times = np.asarray(values, dtype=float)
    require(times.shape == (repeats,), f'Expected {repeats} timing repetitions')
    require(np.isfinite(times).all() and (times > 0).all(), 'Nonpositive or nonfinite timing')
    return float(np.median(times))


def paired_summary(candidate, control, *, seed=20261019, resamples=10000):
    """Resample query pairs; do not treat sessions or repetitions as new queries."""
    require(bool(candidate) and candidate.keys() == control.keys(), 'Unmatched query IDs')
    ids = sorted(candidate)
    a = np.array([candidate[q] for q in ids], dtype=float)
    b = np.array([control[q] for q in ids], dtype=float)
    require(np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all(),
            'Invalid paired query measurements')
    require(isinstance(resamples, int) and resamples > 0, 'Invalid bootstrap count')
    rng = np.random.default_rng(seed)
    changes = []
    for start in range(0, resamples, 256):
        idx = rng.integers(0, len(ids), (min(256, resamples-start), len(ids)))
        changes.extend((a[idx].mean(1)/b[idx].mean(1)-1).tolist())
    return {'candidate_mean_query_median_ms': float(a.mean()),
            'control_mean_query_median_ms': float(b.mean()),
            'relative_change': float(a.mean()/b.mean()-1),
            'query_bootstrap_95ci': np.quantile(changes, [.025,.975]).tolist(),
            'query_wins': int((a < b).sum()), 'query_ties': int((a == b).sum()),
            'query_losses': int((a > b).sum()), 'paired_queries': len(ids),
            'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
            'scope': 'Query resampling only; excludes machine/environment uncertainty. Observed development data.'}


def engineering_summary(data):
    rows = rows_by_query(data)
    require(not data.get('ranking_changes'), 'Recorded ranking changes')
    paired = {m: {} for m in ('ind4','ind8')}
    for qid, row in rows.items():
        methods = row['methods']
        require(methods['ind4']['ids'] == methods['ind8']['ids'], f'Ranking mismatch: {qid}')
        for m in paired:
            z = methods[m]
            require(z['repeat_rank_changes'] == 0, f'Repeated ranking changed: {qid}/{m}')
            require(len(z['ids']) == 100 and len(set(z['ids'])) == 100, f'Invalid Top100: {qid}/{m}')
            require(len(z['scores']) == 100 and np.isfinite(z['scores']).all(), f'Invalid scores: {qid}/{m}')
            paired[m][qid] = median_time(z['times_ms'])
    return {'mean_query_median_ms': {m: float(np.mean(list(paired[m].values()))) for m in paired},
            **paired_summary(paired['ind4'],paired['ind8']), 'queries':len(rows),'passes':5,
            'timing_boundary':'Full existing refinement pipeline; one process session'}


def hybrid_summary(sessions):
    require(len(sessions) == 3, 'Expected three hybrid sessions')
    mapped = [rows_by_query(s) for s in sessions]
    require(all(rows.keys() == mapped[0].keys() for rows in mapped), 'Sessions have different query IDs')
    require(len({s['dataset'] for s in sessions}) == 1, 'Mixed datasets in session pool')
    session_pairs = []
    for rows in mapped:
        paired = {m:{} for m in ('mixed24','local24')}
        for qid, row in rows.items():
            for m in paired:
                metrics = row['methods'][m]
                require(metrics['state_bytes_per_doc'] == 24, f'Unequal state budget: {qid}/{m}')
                require(metrics['upper_violations'] == metrics['strict_missing'] == 0,
                        f'Correctness failure: {qid}/{m}')
                paired[m][qid] = median_time(row['times_ms'][m])
        session_pairs.append(paired)
    pooled = {m:{q:float(np.mean([s[m][q] for s in session_pairs])) for q in mapped[0]}
              for m in ('mixed24','local24')}
    return {'mean_query_median_ms':{m:float(np.mean(list(v.values()))) for m,v in pooled.items()},
            **paired_summary(pooled['local24'],pooled['mixed24']),
            'session_relative_changes':[float(np.mean(list(s['local24'].values())) /
                                              np.mean(list(s['mixed24'].values()))-1) for s in session_pairs],
            'pooling':'Average the three session medians for each query, then resample queries',
            'timing_boundary':'Coarse TopB only, 24B/document for both methods; three sessions'}


def recompute(root):
    root = Path(root)
    result = {'verified_files':verify_manifest(root),'engineering':{},'hybrid':{},
              'scope':'Observed queries; no upstream-system speedup or leaderboard claim'}
    for dataset in ('nq-full','hotpotqa-colbertv2'):
        read = lambda name: json.loads((root/'evidence/raw'/name).read_text())
        result['engineering'][dataset] = engineering_summary(read(f'{dataset}-engineering-full.json'))
        result['hybrid'][dataset] = hybrid_summary([read(f'{dataset}-hybrid-session{i}.json') for i in (1,2,3)])
    return result
