#!/usr/bin/env python3
"""Render archived measurements; no experiments, models or corpus downloads."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maxsim_pruning.evidence import recompute, rows_by_query, median_time, require

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
ap.add_argument('--output',type=Path,default=Path('run-output/report'))
a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
summary=recompute(a.root)
summary['coarse_inputs']={}
datasets={'nq-full':'NQ','hotpotqa-colbertv2':'HotpotQA'}
main={}
for ds in datasets:
    data=json.loads((a.root/f'evidence/raw/{ds}-P512.json').read_text())
    rows=rows_by_query(data)
    require(data['budget']==4096 and data['reps']==5,'Unexpected coarse timing protocol')
    summary['coarse_inputs'][ds]={k:data[k] for k in ['documents','centroids','tokens','dimension','shared_index_bytes']}
    main[ds]={}
    for method in ['ind2','ind4','ind8','cond4','cond8','cond16','delta8','ideal8']:
        metrics=[r['methods'][method] for r in rows.values()]
        require(all(x['upper_violations']==x['strict_missing']==0 for x in metrics),'Incorrect coarse result')
        z={'mean_query_median_ms':float(np.mean([
            median_time(r['methods'][method]['cached_oracle_times_ms'],repeats=3) if method=='ideal8'
            else median_time(r['times_ms'][method]) for r in rows.values()]))}
        z['state_bytes_per_doc']=metrics[0]['state_bytes_per_doc']
        expected_bytes={'ind2':8,'ind4':16,'ind8':32,'cond4':8,'cond8':16,'cond16':32,'delta8':16,'ideal8':16}
        require(z['state_bytes_per_doc']==expected_bytes[method],'Unexpected state budget')
        require(all(x['state_bytes_per_doc']==z['state_bytes_per_doc'] for x in metrics),'Changing state budget')
        for field in ['slack_mean','false_survivors_truecut','false_survivors_seedcut','scored_documents','scored_tokens']:
            z[field]=float(np.mean([x[field] for x in metrics]))
        for field in ['owned_workspace_bytes','workspace_plus_transient_bytes']:
            z[field]=max(x[field] for x in metrics) if all(field in x for x in metrics) else None
        if method=='ideal8':
            z['scope']='Offline oracle; cached time excludes envelope construction, not deployable latency'
            z['offline_envelope_build_ms']=float(np.mean([x['offline_envelope_build_ms'] for x in metrics]))
        phase={q:r['methods'][method] for q,r in rows.items() if 'instrumented_gpu_ms' in r['methods'][method]}
        if method!='ideal8':require(len(phase)==8,'Expected eight separately profiled queries')
        z['profile_query_ids']=sorted(phase)
        if phase:
            z['phase_ms']={f:float(np.mean([x['instrumented_gpu_ms'][f] for x in phase.values()]))
                           for f in next(iter(phase.values()))['instrumented_gpu_ms']}
            z['profile_false_survivors']=float(np.mean([x['false_survivors_truecut'] for x in phase.values()]))
        main[ds][method]=z
    keys=['ind2','ind4','ind8','cond4','cond8','cond16','delta8']
    require(all(main[ds][m]['profile_query_ids']==main[ds]['ind4']['profile_query_ids'] for m in keys),
            'Phase comparison does not use the same query subset')
summary['coarse_diagnostics']=main
(a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})

def save(fig,name):
    fig.savefig(a.output/(name+'.png'),dpi=160)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,(ds,title) in zip(axes,datasets.items()):
    m=main[ds]
    ax.plot([8,16,32],[m[k]['mean_query_median_ms'] for k in ['ind2','ind4','ind8']],'-o',label='Independent terms')
    ax.plot([8,16,32],[m[k]['mean_query_median_ms'] for k in ['cond4','cond8','cond16']],'-s',label='Conditional states')
    ax.set(title=title,xlabel='Packed state bytes / document',ylabel='Mean query-median time (ms)',xticks=[8,16,32])
    ax.grid(alpha=.2);ax.legend()
fig.suptitle('State budget and total coarse Top4096 latency')
fig.text(.5,.02,'64 fixed queries, 5 paired passes. State bytes exclude the rest of the workspace. No raw refinement.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.07,1,.94]);save(fig,'state-budget')

fig,axes=plt.subplots(2,2,figsize=(11,8))
keys=['ind2','ind4','ind8','cond4','cond8','cond16','delta8']
colors=plt.cm.tab10(np.arange(len(keys)))
for row,(ds,title) in enumerate(datasets.items()):
    m=main[ds]
    for k,color in zip(keys,colors):
        z=m[k]
        axes[row,0].scatter(z['slack_mean'],z['false_survivors_truecut']/1e3,color=color,label=k)
        score_ms=z['phase_ms']['seed_score']+z['phase_ms']['survivor_score']
        axes[row,1].scatter(z['profile_false_survivors']/1e3,score_ms,color=color,label=k)
    axes[row,0].set(title=title+' | formal 64 queries',xlabel='Mean upper-bound slack',ylabel='False survivors (thousands)')
    axes[row,1].set(title=title+' | same 8 profiled queries',xlabel='False survivors (thousands)',ylabel='Candidate scoring GPU time (ms)')
    for ax in axes[row]:ax.grid(alpha=.2)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=7,bbox_to_anchor=(.5,.04),frameon=False)
fig.text(.5,.015,'False survivor: upper >= true TopB cutoff AND true coarse score < cutoff. Ties are not false survivors.',ha='center',fontsize=9)
fig.suptitle('Bound slack, ineffective survivors and scoring cost')
fig.tight_layout(rect=[0,.09,1,.94]);save(fig,'bound-cost')

groups={'LUT + topP':['lut_top'],'Selection + table':['selection_table'],
        'Encode + scatter':['encode_tiles','scatter'],'Decode + seed':['decode','seed_select'],
        'Candidate scores':['seed_score','survivor_score'],'Compact + finish':['threshold_compact','final_select']}
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,(ds,title) in zip(axes,datasets.items()):
    methods=['ind4','ind8','cond8'];bottom=np.zeros(3)
    for label,fields in groups.items():
        vals=[sum(main[ds][m]['phase_ms'][f] for f in fields) for m in methods]
        ax.bar(methods,vals,bottom=bottom,label=label);bottom+=vals
    ax.set(title=title,ylabel='Instrumented GPU phase time (ms)');ax.grid(axis='y',alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.045),frameon=False)
fig.suptitle('Stage costs on the same 8 profiled queries')
fig.text(.5,.015,'Profiling excludes CPU overhead and uses n=8; phase sums are not the formal n=64 wall-clock totals.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.16,1,.94]);save(fig,'stages')

def table(section):
    out='<table><tr><th>Dataset</th><th>Time change</th><th>Query bootstrap 95% interval</th><th>Wins / ties / losses</th></tr>'
    for ds,z in summary[section].items():
        lo,hi=z['query_bootstrap_95ci']
        out+=f'<tr><td>{datasets[ds]}</td><td>{z["relative_change"]:+.2%}</td><td>[{lo:+.2%}, {hi:+.2%}]</td><td>{z["query_wins"]} / {z["query_ties"]} / {z["query_losses"]}</td></tr>'
    return out+'</table>'

body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MaxSim evidence review</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#243746}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde4ea}aside{background:#edf4f8;padding:18px}a{color:#1463a0}</style><h1>MaxSim evidence review</h1><aside>Observed development queries. These are matched own-control comparisons, not public-system or SOTA claims. Query bootstrap excludes environment variability. Archived evidence is hash verified before rendering.</aside><h2>Independent 4-bin vs 8-bin control</h2><p>Full existing refinement pipeline; 64 paired queries, five passes, one process session. Negative time change means faster.</p>'''
body+=table('engineering')+'<h2>Conditional hybrid vs independent mixed-bin control</h2><p>Coarse TopB only; both 24B/document. Average three session medians within each query before resampling. The positive mean change supports stopping the current hybrid route.</p>'+table('hybrid')
for name,title in [('state-budget','State budget'),('bound-cost','Bound quality and scoring cost'),('stages','Stage decomposition')]:
    body+=f'<h2>{title}</h2><img src="{name}.png" alt="{title}">'
body+='<h2>Accounting and exclusions</h2><p>Packed state sizes are not total memory. Per-method workspace maxima and shared-scope diagnostics are in <a href="summary.json">summary.json</a>. Cached ideal tables are offline oracles and are excluded from deployable latency curves. Operational seed cutoffs may differ from true TopB cutoffs. Failed/invalid configurations remain in evidence/failed and are excluded from successful aggregates.</p></html>'
(a.output/'index.html').write_text(body)
print(a.output.resolve())
