"""Use the same measurement/audit harness, with one fixed local hybrid and24B control."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'v6_diagnose.py').read_text()
s=s.replace("p.add_argument('--pilot',action='store_true');", "p.add_argument('--session',type=int,default=1);p.add_argument('--pilot',action='store_true');")
s=s.replace("f'{ds}-P{a.prefix}{\"-pilot\"if a.pilot else \"\"}.json'", "f'{ds}-hybrid-session{a.session}{\"-pilot\"if a.pilot else \"\"}.json'")
start=s.index('spec=[');end=s.index("ref=torch.empty",start)
s=s[:start]+'''from v6_hybrid_engine import Hybrid24
kw=dict(programs=8192,block=256,compact_final=False,unsorted_seed=True)
engines={'ind4':DiagnosticFrontier(cent,codes,off,io,ivf,budget,512,2,4,groups=32,score_power=2.,kind='independent',**kw),'mixed24':Hybrid24(cent,codes,off,io,ivf,budget,512,2,8,merge=False,**kw),'local24':Hybrid24(cent,codes,off,io,ivf,budget,512,2,8,merge=True,**kw)}
'''+s[end:]
start=s.index(' # Complete finite-codebook');end=s.index(' rows.append(row)',start);s=s[:start]+s[end:]
s=s.replace("rng=np.random.default_rng(2026102001)","rng=np.random.default_rng(2026102001+a.session)")
s=s.replace("row['methods'][name]=z", "row['methods'][name]=z\n  if name=='local24':z['online_guide']=e.anchors[e.assign.long()].cpu().tolist()")
s=s.replace('Observed-data coarseTopB diagnosis. CPU embedding to CPU B IDs; no token refinement. Whole shuffled paired passes after explicit warmup. Oracle is offline and its build cost excluded only in explicitly cached_oracle times. Instrumented GPU events separate from formal wallclock.', 'One fixed online8pair policy; fullychargedselection/table/packing/scoring; matched24B independentmixedbins plus16Bind4. Three independently restarted measuredsessions; observeddata,noheldoutclaim. CPUembeddingtoCPUcoarseTopB; no downstreamtokenrefinement. Instrumentsseparate.')
exec(compile(s,str(R/'v6_diagnose.py'),'exec'))
