#!/usr/bin/env python3
"""Verify archived hashes and recompute published engineering comparisons."""
import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def recompute():
    manifest = json.loads((ROOT/'evidence/SHA256SUMS.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == entry['sha256'], name
    result = {'verified_files': len(manifest), 'engineering': {}, 'hybrid': {},
              'scope': 'Observed queries; not an upstream benchmark or leaderboard. Query repetitions are not independent queries.'}
    for dataset in ('nq-full', 'hotpotqa-colbertv2'):
        d = json.loads((ROOT/f'evidence/raw/{dataset}-engineering-full.json').read_text())
        assert d['complete'] and len(d['records']) == 64 and not d['ranking_changes']
        for row in d['records']:
            assert row['methods']['ind4']['ids'] == row['methods']['ind8']['ids']
            for method in row['methods'].values():
                assert len(method['times_ms']) == 5 and method['repeat_rank_changes'] == 0
        times = {m: mean(median(r['methods'][m]['times_ms']) for r in d['records']) for m in ('ind4','ind8')}
        result['engineering'][dataset] = {'mean_query_median_ms': times,
            'relative_change': times['ind4']/times['ind8']-1, 'queries':64, 'passes':5,
            'scope':'Full existing refinement pipeline, one process session'}
        sessions = [json.loads((ROOT/f'evidence/raw/{dataset}-hybrid-session{i}.json').read_text()) for i in (1,2,3)]
        for session in sessions:
            assert session['complete'] and not session['failures'] and len(session['records']) == 64
        times = {m: mean(median(r['times_ms'][m]) for d in sessions for r in d['records']) for m in ('mixed24','local24')}
        result['hybrid'][dataset] = {'mean_query_median_ms': times,
            'relative_change': times['local24']/times['mixed24']-1, 'scope':'Coarse TopB only, three sessions; 24B per document for both'}
    return result


if __name__ == '__main__':
    print(json.dumps(recompute(), indent=2))
