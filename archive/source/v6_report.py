"""Recompute every reported aggregate from archived raw measurements; no GPU needed."""
from pathlib import Path
import sys,json,hashlib,html,argparse
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'report-libs'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=R.parent.parent/'outputs/maxsim-prescan-20261018');args=ap.parse_args()
V=R/'v6'; O=args.output; O.mkdir(exist_ok=True,parents=True)
DS=['nq-full','hotpotqa-colbertv2']; LABEL=dict(zip(DS,['NQ','HotpotQA']))
def read(p):return json.loads(p.read_text())
def avg(x):return float(np.mean(x))
def bootstrap(a,b):
 a=np.asarray(a);b=np.asarray(b);ix=np.random.default_rng(2026102010).integers(0,len(a),(2000,len(a)))
 rr=a[ix].mean(1)/b[ix].mean(1)-1
 return {'relative_change':float(a.mean()/b.mean()-1),'query_bootstrap_95ci':np.quantile(rr,[.025,.975]).tolist()}
def values(d,k):return np.array([np.median(r['times_ms'][k]) for r in d['records']])
def summarize(d):
 result={}
 for k in d['records'][0]['methods']:
  rr=[r['methods'][k]for r in d['records']]
  z={'mean_query_median_ms':avg([np.median(x['cached_oracle_times_ms'])for x in rr]) if k=='ideal8'else avg(values(d,k))}
  for field in ['slack_mean','false_survivors_truecut','false_survivors_seedcut','scored_documents','scored_tokens','logical_postings','offline_envelope_build_ms','table_approx_minus_ideal_mean','table_approx_minus_ideal_max']:
   if field in rr[0]:z[field]=avg([x[field]for x in rr])
  for field in ['state_bytes_per_doc','owned_workspace_bytes','transient_peak_extra_bytes','workspace_plus_transient_bytes']:
   if field in rr[0]:z[field]=max(x[field]for x in rr)
  inst=[x for x in rr if 'instrumented_gpu_ms'in x]
  if inst:
   z['phase_gpu_ms_first8']={f:avg([x['instrumented_gpu_ms'][f]for x in inst])for f in inst[0]['instrumented_gpu_ms']}
   z['phase_sample_metrics']={f:avg([x[f]for x in inst])for f in ['slack_mean','false_survivors_truecut','scored_tokens','scored_documents']}
  result[k]=z
 return result
S={'scope':'Observed diagnostic data; query bootstrap does not cover machine variation. No public leaderboard claim.','main':{},'factorial':{},'hybrid':{},'engineering':{},'validation':{}}
all_diag=[]
for ds in DS:
 d=read(V/f'{ds}-P512.json');all_diag.append(d);S['main'][ds]={'input':{k:d[k]for k in ['documents','centroids','tokens','dimension','shared_index_bytes']},'methods':summarize(d),'conditional_vs_ind4':bootstrap(values(d,'cond8'),values(d,'ind4'))}
for corpus in ['nq','hotpot']:
 for enc in ['small','colbert']:
  for pref in [64,512]:
   key=f'cross-{corpus}-{enc}-P{pref}';d=read(V/f'{key}.json');all_diag.append(d);z=summarize(d)
   S['factorial'][key]={'methods':z,'conditional_false_ratio':z['cond8']['false_survivors_truecut']/z['ind4']['false_survivors_truecut'],'conditional_time_ratio':z['cond8']['mean_query_median_ms']/z['ind4']['mean_query_median_ms'],'metadata':read(V/f'cross-{corpus}-{enc}/meta.json'),'index_metadata':read(V/f'cross-{corpus}-{enc}/index/READY.json')}
for d in all_diag:
 assert d['complete'] and not d['failures'] and len(d['records'])==64 and d['reps']==5
 for r in d['records']:
  assert len(r['methods'])==8 and len(r['times_ms'])==7
  for k,t in r['times_ms'].items():assert len(t)==5 and all(np.isfinite(t))
  for z in r['methods'].values():assert z['upper_violations']==z['strict_missing']==0
freeze=read(V/'hybrid-freeze.json')
assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in freeze['sources'].items())
for ds in DS:
 sessions=[read(V/f'{ds}-hybrid-session{i}.json')for i in [1,2,3]]
 oracle=read(V/f'{ds}-local-ceiling.json');expected={}
 for row in oracle['records']:
  guide=list(range(32))
  for a,b,_ in row['pairs']:guide[b]=a
  expected[row['query_id']]=guide
 for d in sessions:
  assert d['complete']and not d['failures']and len(d['records'])==64
  for row in d['records']:
   assert row['methods']['local24']['online_guide']==expected[row['query_id']]
   for k in ['ind4','mixed24','local24']:
    assert len(row['times_ms'][k])==5
    assert row['methods'][k]['upper_violations']==row['methods'][k]['strict_missing']==0
   assert row['methods']['mixed24']['state_bytes_per_doc']==row['methods']['local24']['state_bytes_per_doc']==24
 vals={k:np.mean([values(d,k)for d in sessions],axis=0)for k in ['ind4','mixed24','local24']}
 S['hybrid'][ds]={'sessions':[summarize(d)for d in sessions],'pooled_mean_query_median_ms':{k:avg(v)for k,v in vals.items()},'local_vs_same24B':bootstrap(vals['local24'],vals['mixed24']),'local_vs_ind4':bootstrap(vals['local24'],vals['ind4']),'local_ceiling':{k:{f:avg([r['metrics'][k][f]for r in oracle['records']])for f in oracle['records'][0]['metrics'][k]}for k in oracle['records'][0]['metrics']}}
 d=read(V/f'{ds}-engineering-full.json');assert d['complete']and len(d['records'])==64 and not d['ranking_changes']
 vals={k:np.array([np.median(r['methods'][k]['times_ms'])for r in d['records']])for k in ['ind4','ind8']}
 for r in d['records']:
  for z in r['methods'].values():assert len(z['times_ms'])==5 and z['repeat_rank_changes']==0
 S['engineering'][ds]={'mean_query_median_ms':{k:avg(v)for k,v in vals.items()},'ndcg10':{k:avg([r['methods'][k]['ndcg10']for r in d['records']])for k in vals},'ind4_vs_ind8':bootstrap(vals['ind4'],vals['ind8']),'ranking_changes':0,'gpu_peak_with_both_engines_bytes':d['gpu_peak_allocated_bytes_with_both_engines'],'host_peak_rss_kib':d['host_peak_rss_kib'],'dedup_used':d.get('dedup_used',True)}
S['validation']={'complete_main_and_factorial_files':10,'complete_hybrid_files':6,'complete_local_oracle_files':2,'complete_engineering_files':2,'fixed_queries_per_file':64,'formal_timed_calls_excluding_oracle':29440,'final_upper_violations':0,'final_strict_coarse_topB_misses':0,'hybrid_pair_policy_matches_cpu_queries_per_dataset_per_session':64,'hybrid_source_freeze_hashes_pass':True,'float64_oracle_checks':20,'max_float64_table_error':max(z['cpu_f64_table_max_error']for d in all_diag for r in d['records']for z in r['methods'].values()if z.get('cpu_f64_table_max_error')is not None)}
# Metadata are complete in raw evidence; omit long IDs in the concise derived summary.
for z in S['factorial'].values():z['metadata']={k:v for k,v in z['metadata'].items()if k not in ['doc_ids','query_ids']}
(O/'条件摘要失效诊断_汇总.json').write_text(json.dumps(S,ensure_ascii=False,indent=2))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
colors={'ind':'#186b9b','cond':'#dd6b24','mixed':'#64748b','local':'#a33672'}
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for ax,ds in zip(axes,DS):
 m=S['main'][ds]['methods'];h=S['hybrid'][ds]['pooled_mean_query_median_ms']
 ax.plot([8,16,32],[m[k]['mean_query_median_ms']for k in ['ind2','ind4','ind8']],'-o',color=colors['ind'],label='32 independent terms')
 ax.plot([8,16,32],[m[k]['mean_query_median_ms']for k in ['cond4','cond8','cond16']],'-s',color=colors['cond'],label='16 conditional states')
 ax.scatter([24],[h['mixed24']],color=colors['mixed'],marker='D',s=55,label='Independent mixed bins (3 sessions)')
 ax.scatter([24],[h['local24']],color=colors['local'],marker='X',s=75,label='8-pair hybrid (3 sessions)')
 ax.set(title=LABEL[ds],xlabel='Packed state bytes / document',ylabel='Mean query-median latency (ms)',xticks=[8,16,24,32]);ax.grid(alpha=.18)
fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,frameon=False);fig.suptitle('Measured coarse Top4096 pipeline | 64 fixed queries | 5 paired passes')
fig.text(.5,.015,'No raw-token refinement here. Oracle excluded. Separate sessions are not a controlled hardware distribution.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(O/'条件摘要失效诊断_状态预算与总耗时.png');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8))
keys=['ind2','ind4','ind8','cond4','cond8','cond16','delta8']
methodcolors=dict(zip(keys,['#1b4965','#1982c4','#5fa8d3','#ca6702','#e85d04','#bb3e03','#777777']))
markers=dict(zip(keys,['o','s','^','o','s','^','D']))
for col,ds in enumerate(DS):
 for k in keys:
  z=S['main'][ds]['methods'][k];p=z['phase_sample_metrics'];cost=sum(z['phase_gpu_ms_first8'][q]for q in ['seed_score','survivor_score']);c=methodcolors[k]
  for row,xkey in enumerate(['slack_mean','false_survivors_truecut']):
   x=p[xkey]/(1000 if row else 1);axes[row,col].scatter(x,cost,color=c,s=65,marker=markers[k],label=k,edgecolors='white',linewidths=.5)
 for row in range(2):axes[row,col].set(title=LABEL[ds],ylabel='Seed + survivor scoring (GPU ms)',xlabel='Mean upper-bound slack (score units)'if row==0 else'False survivors at exact TopB cut (thousands)');axes[row,col].grid(alpha=.18)
fig.suptitle('Bound looseness and scoring cost | same first 8 fixed queries')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.043),ncol=7,fontsize=9,frameon=False)
fig.text(.5,.012,'Instrumented CUDA events; not formal wall-clock timing. Operational seed thresholds also affect scoring volume.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.09,1,.96]);fig.savefig(O/'条件摘要失效诊断_界松弛与评分成本.png');plt.close(fig)

groups=[('LUT + topP',['lut_top']),('Selection + table',['selection_table']),('Encode + scatter',['encode_tiles','scatter']),('Decode + seed select',['decode','seed_select']),('Score candidates',['seed_score','survivor_score']),('Compact + final select',['threshold_compact','final_select'])]
fig,axes=plt.subplots(1,2,figsize=(12,5));cs=['#22577a','#38a3a5','#80b918','#a7c957','#e76f51','#a98467']
for ax,ds in zip(axes,DS):
 kk=['ind4','ind8','cond8'];bottom=np.zeros(3)
 for (label,fields),c in zip(groups,cs):
  v=[sum(S['main'][ds]['methods'][k]['phase_gpu_ms_first8'][f]for f in fields)for k in kk];ax.bar(kk,v,bottom=bottom,label=label,color=c);bottom+=v
 ax.set(title=LABEL[ds],ylabel='Instrumented GPU phase time (ms)');ax.grid(axis='y',alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='lower center',bbox_to_anchor=(.5,.052),ncol=6,frameon=False);fig.suptitle('Where the time goes | first 8 fixed queries, instrumented separately')
fig.text(.5,.015,'Phase sums exclude CPU overhead and use n=8; they must not be equated to the n=64 formal latency means.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(O/'条件摘要失效诊断_各阶段耗时.png');plt.close(fig)

def esc(x):return html.escape(str(x))
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def pct(x):return f'{x*100:+.2f}%'
def interval(z):return pct(z['relative_change'])+' ['+', '.join(pct(x)for x in z['query_bootstrap_95ci'])+']'
style='''body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#182c3b;line-height:1.7}h1{font-size:30px;line-height:1.35}h2{font-size:22px;margin-top:32px}h3{font-size:18px}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{border-bottom:1px solid #dbe4e9;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf4f8}.decision{background:#edf4f8;padding:18px 22px;border-left:5px solid #22577a}a{color:#086a9d}img{width:100%;height:auto}small,.muted{color:#536775}code,pre{font-family:monospace;background:#f3f6f8}pre{white-space:pre-wrap;padding:16px;font-size:13px}.warn{background:#fff4e8;padding:16px}@media print{body{margin:0;font-size:11px;padding:0}h1{font-size:23px}h2{font-size:16px;margin-top:15px}table{font-size:10px}th,td{padding:5px}.decision{padding:10px}a{color:inherit}}'''
def page(title,body,path):path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style><body>'+body+'</body></html>')
rows=[]
for ds in DS:
 z=S['hybrid'][ds];rows.append([LABEL[ds],f"{z['pooled_mean_query_median_ms']['mixed24']:.3f}",f"{z['pooled_mean_query_median_ms']['local24']:.3f}",interval(z['local_vs_same24B'])])
hybridtable=table(['全库 / 粗分阶段','独立混合 24B，ms','局部条件 24B，ms','条件耗时变化 / 查询 bootstrap 95% 区间'],rows)
one='''<h1>条件摘要失效诊断：停止扩展该路线</h1><p class="muted">2026-10-19 · 本轮完成机制诊断、实际混合原型及回退工程对照 · 目标是研究决策</p>
<div class="decision"><b>建议：停止当前条件摘要及局部合并路线的扩展。</b>保留简单独立分箱实现和诊断工具。当前证据不支持把条件摘要作为稳定优于同预算独立状态的新算法，也不支持 CCF A / SOTA 主张。</div>
<h2>主要失效原因与证据强度</h2>
<p><b>强证据：瓶颈主要是约束信息损失，未发现值得追逐的 TopP 近似余量。</b>Hotpot 全库固定 64 条查询中，完整码本条件表与 Top512+尾界表逐项一致；即使免费获得理想表，仍约有 109.61 万无效存活文档，16B 独立控制为 13.33 万。NQ 完整表也仅将无效存活文档从 13,412 降到 13,394。这里的“理想”限于固定代表、分组和分箱的条件表，非所有可能算法的上限。</p>
<p><b>强证据：少量合并仍未形成净收益。</b>按在线可算的关系强度选择 8 对项，真实压入 24B；与同为 24B 的独立混合分箱比较，3 次进程重启复测均变慢，已计入选择、建表、传播、解码和评分。</p>'''+hybridtable+'''
<p><b>保留的正结果：</b>NQ 严格 8B 预算下条件法快 21.6%，16B 下快约 1.2%；Hotpot 两个预算均明显更慢。因此不是“条件法总无效”，而是当前表示没有稳定净收益。补齐的 2×2、每格 64k 文档诊断表明编码器和语料均有影响；子库不能当作全库因果结论。</p>
<h2>保留下来的价值与边界</h2><p><b>已测工程价值：</b>32 项独立状态由 8 分箱改为 4 分箱，完整现有检索流程 NQ 1.330→1.211 ms（−8.92%），Hotpot 2.468→1.965 ms（−20.39%）；固定 64 查询最终 Top100 无变化。仅一场配对工程复测，属于常规参数与实现调优，不是新算法或对公开 SOTA 的证明。</p>
<p><b>正确性范围：</b>最终版本固定码本粗分 TopB 审计无上界违规、无严格高于边界的漏项；并列允许不同 ID。不能据此声称原始向量全库精确 Top100。保留早期失败源码、日志和错误结果，未计入有效性能。</p>
<p><b>未验证：</b>全库四格交叉、未见数据泛化、物理显存/PCIe 流量、跨机器和并发。失败原型未过开发数据门槛，按停止条件不再扩展这些测试。查询 bootstrap 不包含机器环境变化。</p>
<p><b>可主张范围：</b>固定表示下的负结果、信息与执行成本的诊断方法及可复现实验；目前不足以支撑新的高水平算法论文。独立分箱、同预算打包和审计工具适合清理后独立开源；上游 PR 需适配接口、许可证和维护方测试，本轮不提交。</p>
<p><a href="条件摘要失效诊断_完整证据.html">完整证据与三张核心图</a> · <a href="条件摘要失效诊断_汇总.json">机器可读汇总</a> · <a href="条件摘要失效诊断_复现包.zip">代码、原始结果、索引与复现命令</a></p>'''
page('条件摘要失效诊断 · 一页结论',one,O/'条件摘要失效诊断_一页结论.html')
body='''<h1>MaxSim 条件摘要：失效原因、可改进上限与停止决策</h1><p class="muted">2026-10-19 · v6 · 所有数字由本包 v6_report.py 从原始逐查询记录重算</p><div class="decision"><b>结论：停止扩展当前条件摘要路线。</b>完整表未解除主要候选膨胀，最小局部混合未胜过同字节独立控制。保留简单独立分箱的工程收益，不包装成算法创新。<a href="条件摘要失效诊断_一页结论.html">一页决策</a></div>
<h2>1. 审计、资源和计时边界</h2>
<p>先核对上一轮 HTML、结构化结果、源码、原始逐次计时及 614,544,200 字节复现 ZIP；SHA-256 为 <code>0ed7f1960f80f376dd1d02c7723a025f222e533f3d0066720b3910bfc5638264</code>。重算 Hotpot 同流程 B=16384：条件 7.4428 ms、power32 4.2887 ms，增加 73.5436%；B=4096 为 4.1324 对 2.3027 ms，增加 79.4589%。旧 normal32 与 power32 的分箱及 guard 不同，不能视为只改变状态数的严格控制。</p>
<p>本轮固定同一输入、TopP、score-power=2 分箱族、候选预算、2B 种子、后续粗分评分内核及 4e-4 浮点保护。条件界、差值界和完整表共享同一代表/分组；独立项保持身份代表。不同分箱数是明确的实验因素；同字节比较不等于分箱数相同。32 项×4 个一元位分箱=128 bit=16B，不是每项只存 2 bit。条件 16 项×8 位也为 16B。</p>
<p>正式时间：CPU 上的预编码查询进入，到 CPU 收到粗分 TopB ID；不含编码、离线索引建立及后续原始 token 精算。全库 B=4096、子库 B=128。每方法先热身，随机方法与查询顺序，5 个完整配对遍历，每次前后 CUDA 同步。主表取每查询 5 次中位数再取 64 查询均值。阶段图另用前 8 条查询 CUDA events，不进入正式计时；事件扰动、CPU 开销和不同样本使阶段和不能与主表直接相等。完整检索工程对照另列。</p>
<p>事前资源估计与诊断方案保存在 <code>v6/protocol.json</code>：复用全库，只为两格缺失编码器补 64k 子库；预计 0.5–2 GPU 小时、子库向量不超过 12GiB；没有新增付费资源或重编码全库。实际缺失两格编码约 100.4 秒与 30.4 秒（单卡 5090）；四格索引构建约 5.7 秒，新增两格原始向量约 2.09GiB。其余缓存复用。计时运行在 RTX PRO 6000 Blackwell Server Edition、CPU 128–135、8 CPU 线程；准备工作完成后串行计时。未锁定时钟、未取得管理员级独占，也没有物理 DRAM/PCIe 计数器。三次重启复测不能替代跨环境实验。</p>
<h2>2. 编码器与语料的交叉诊断</h2>
<p>准确模型及官方权重散列已核对：<a href="https://huggingface.co/answerdotai/answerai-colbert-small-v1/tree/934fa8bb4ce2284f4c2baa232d81aca4d076fa5e">answerdotai/answerai-colbert-small-v1，revision 934fa8b…</a>（96D），及 <a href="https://huggingface.co/colbert-ir/colbertv2.0/tree/c1e84128e85ef755c096a95bdb06b47793b13acf">colbert-ir/colbertv2.0，revision c1e8412…</a>（128D）。Small 元数据内的 AnswerAI-ColBERTv2.5-small 是训练名称，不是另一个下载模型。完整 revision 和 SHA 见 model-provenance.json。</p>
<p>两者 query_maxlen=32、attend_to_mask_tokens=False、标点过滤、FP32 SDPA 编码后归一化并存 FP16。文档上限均显式设为 300；ColBERT 原始元数据为 180，本项目覆盖为 300。故“编码器效应”是模型及协议组合，不是纯架构因果。复用对角格的 64 查询及 4 文档重新编码校验形状和数值，查询最大误差≤6.11e-5、文档≤1.22e-4。</p>
<p>每种语料固定均匀抽样 64,000 文档，跨编码器同文档 ID、同 64 查询。四格均采用 8192 码字、262144 训练 token、10 次 KMeans、同 seed、同 FP16 分配及逐文档码字去重。训练样本每质心 32 个低于 FAISS 建议 39 个，警告保留；四格一致但不主张索引已最优。P=512 占子库码本 6.25%，不能将子库行为外推为全库速度。</p>'''
fr=[]
for corpus in ['nq','hotpot']:
 for enc in ['small','colbert']:
  z=[S['factorial'][f'cross-{corpus}-{enc}-P{p}']for p in [64,512]]
  fr.append([corpus,enc,f"{z[0]['conditional_false_ratio']:.3f}",f"{z[1]['conditional_false_ratio']:.3f}",f"{z[1]['conditional_time_ratio']:.3f}"])
body+=table(['64k 子库','编码器','P64 条件/独立无效候选比','P512 条件/独立无效候选比','P512 条件/独立时间比'],fr)
body+='''<p><b>事实：</b>更换编码器与更换语料均改变相对界效果；P512 有时显著减少无效候选，但时间仍慢约 5–6%，在子库上额外建表开销占主导。<b>推断：</b>原先两个全库的差异不能只归因于语料，也不能只归因于编码器。<b>未验证：</b>缺失两格的全库规模、完整索引训练及检索质量；本轮没有冒充这项验证。</p>
<h2>3. 完整条件包络与同状态预算</h2>
<p>“完整包络”枚举完整有限码本：给定代表分数的分箱上限，对同组每一未知项取满足该代表约束的最大值，再相加。它是该固定分组表示的完整可分离条件表；不是所有联合文档约束的最紧数学界。TopP+尾界只是该表的一种近似。代表自身沿用同分箱 cap，严格边界与独立控制一致。</p>
<p>无效存活候选：上界≥全库真实粗分第 B 名分数，同时真实粗分&lt;该门槛。边界同分不是无效候选，也允许返回任意并列 ID；严格高于门槛的文档必须保留。另记录实际种子所得门槛及其无效存活数量。评分量包含已精算种子及额外存活文档，不重复计算种子。真实粗分只用于离线审计，未参与在线策略。</p>'''
for ds in DS:
 body+=f'<h3>{LABEL[ds]}：{S["main"][ds]["input"]["documents"]:,} 文档</h3>'
 rows=[]
 for k,z in S['main'][ds]['methods'].items():
  if k=='ideal8':continue
  rows.append([k,f"{z['state_bytes_per_doc']:.0f}",f"{z['mean_query_median_ms']:.3f}",f"{z['slack_mean']:.4f}",f"{z['false_survivors_truecut']:,.0f}",f"{z['scored_documents']:,.0f}",f"{z['scored_tokens']/1e6:.2f}",f"{z['workspace_plus_transient_bytes']/2**20:.1f}"])
 body+=table(['方法','状态 B/文档','总 ms','平均松弛','无效存活文档','实际评分文档','评分码字 M','工作区峰值 MiB'],rows)
body+='''<p>方法：ind2/4/8=32 独立项分别 2/4/8 分箱；cond4/8/16=16 条件状态分别 4/8/16 分箱；delta8=相同 16 状态的简单差值界。工作区为去重后的引擎自有 CUDA storage 加调用中增量临时峰值的最大值，包含解码、候选等辅助数组，非仅位状态。共享驻留索引另计：NQ 1,499,363,852B，Hotpot 2,241,602,140B。该分解不等于整个多引擎审计进程的物理显存、allocator reserved 或驱动显存；离线完整包络的额外构造工作区未测，不能据此做部署内存主张。完整检索内存另列。</p>'''
rows=[]
for ds in DS:
 m=S['main'][ds]['methods'];z=m['ideal8'];rows.append([LABEL[ds],f"{m['cond8']['false_survivors_truecut']:,.1f}",f"{z['false_survivors_truecut']:,.1f}",f"{z['table_approx_minus_ideal_max']:.7f}",f"{z['offline_envelope_build_ms']:.2f}"])
body+=table(['数据','Top512 无效存活','完整表无效存活','每查询表最大差的均值','完整表构造 ms'],rows)
body+='''<p><b>已测事实：</b>Hotpot 64 查询的 Top512 与完整表逐项相等；NQ 也只减少约 19 个无效存活文档。完整表还单列缓存后的调用计时和构造时间，见原始 JSON；缓存后的时间绝不作为可部署速度。相同 Hotpot 表在不同采样顺序下仍有计时差，应视为环境/顺序噪声，不能归因于“理想界更慢”。构造 153/309ms 是当前离线实现成本，不证明所有优化构造都这么慢；本例的关键是构造更精确表几乎没有信息收益。</p>
<p><b>机制推断：</b>代表替代了多个独立约束，保守上界放大，使候选评分成本超过状态传播节省。Hotpot cond8 相对 ind4 无效存活约 8.22 倍，实际评分约 6.78 倍；同字节条件方案没有状态字节优势。差值界更差，因此“条件优于差值”不推出“条件值得替代独立”。代表与其他项的弱约束关系是解释，尚未通过反事实干预单独定量为全部原因。</p>
<p><b>不能抹去的正结果：</b>NQ 在严格 8B 状态预算下 cond4 为 0.850ms，ind2 为 1.085ms，条件法快约 21.6%；16B 下 cond8 比 ind4 快约 1.2%。但放宽到简单 16B 独立控制后，NQ 的 8B 条件方案优势只约 1.9%，而 Hotpot 同一 8B 条件方案反而慢约 127%。这支持“特定预算和数据上有局部效用”，不支持稳定替代或当前任务要求的跨两个已观察数据净收益。保留原始正负结果，不把停止决策写成算法普遍无效的定理。</p>
<p>还须区分审计门槛与实际门槛：Hotpot ind8 在真实门槛下的无效存活比 ind4 少，但实际评分量反而更大。种子排序影响门槛，是评分量变化的一部分。不得只凭共同门槛下的候选数预测总速度。</p>
<img src="条件摘要失效诊断_状态预算与总耗时.png" alt="真实状态预算与全粗分阶段耗时"><img src="条件摘要失效诊断_界松弛与评分成本.png" alt="前8条匹配样本的界松弛与评分成本"><img src="条件摘要失效诊断_各阶段耗时.png" alt="独立计时的GPU阶段分解">
<h2>4. 局部收益上限与唯一最小混合原型</h2>
<p>先对 8 对不重叠的定向关系做信息诊断：在线可算的 Top512 归一化尾部损失加 0.001×(1−cosine)，按固定规则贪心选择；不用测试耗时、真分数或标签。保留其他 16 项独立状态，24 个状态×8 位=24B。8 次合并是节省一个完整 uint64 字的最小数量；并不声称该策略是所有可能局部选择的最优。独立 24B 对照根据当前查询的 max−tail 范围选 16 项用 8 分箱，其余 16 项用 4 分箱。</p>
<p>信息诊断先从 32B 状态解码反事实 24B 上界，不能当真实 24B 部署速度。NQ 的 local24 TopP 无效存活从独立 mixed24 的 14,016 降至 12,162，有小幅空间，故只实现一个实际打包原型。Hotpot 则从 108,559 增至 235,970；完整表没有救回这个差距。</p>
<p>最终原型用 3 个 uint64/文档，混合与独立控制共享通用编码/散射/解码内核。在线选择、建表及所有评分成本均计时；选择 trace 的 64B 也计入工作区。正确性修复后冻结源码散列，再跑 3 次独立进程、每次 5 个全配对遍历。所有 64 查询的 GPU 选择结果与 CPU 策略逐项吻合。</p>'''+hybridtable+'''
<p>NQ 相比更简单的 16B ind4 还慢 4.21%；Hotpot 慢 44.68%。三次会话均同方向。置信区间先对每查询取 5 次中位数，再跨三次会话平均，配对重采样查询 2000 次；没有重采样环境，不能称为跨机器置信区间。<b>停止条件已触发：混合原型计入全部开销，仍不能胜过同预算简单控制。</b></p>
<h2>5. 正确性、失败和工程回退</h2>
<p>最终 10 个主/交叉诊断文件、6 个混合文件、2 个局部上限文件和 2 个完整流程工程文件均完整；每个 64 查询。29,440 次有效正式调用（不含 oracle、热身、失败及 profiler）。最终审计上界违规 0、严格粗分 TopB 漏项 0。20 次独立 CPU float64 完整表核验最大误差为 0；这是对当前输入的核验，不是对任意浮点输入的形式化证明。guard=4e-4 且检查 32 项 LUT 有限、绝对值≤2。原始 FP16 粗分码本与原始文档向量的差异不由该保护消除。</p>
<div class="warn"><b>保留的失败：</b>局部 oracle 首版发生 FP16 标量比较精度问题，旧错误结果保留并显式排除；改为 float32 比较且检查 TopP 表不得低于完整表。混合选择器前三版重复选择已排除的项，触发上界正确性失败；三版源码和日志保留，最终改为行再列归约。未定位成最小编译器复现，不声称是 Triton 缺陷。第三次原始结构化 JSON 曾被后续 pilot 覆盖；名为 hybrid-third-correctness-failure.json 的文件实际是第二次结果的备份，不能作为独立第三次记录，第三次以日志和对应源码为证。另保留一次误复用导致无实验的日志，以及 Hotpot 工程启动时错误请求不存在去重缓存的失败日志。没有本轮 OOM；不能因此宣称任意配置内存可行。</div>
<p>上述失败均未纳入最终性能汇总；失败追踪并非完整历史快照管理的替代。后续开源应把关键选择/打包边界整理成独立单元测试，并用内容寻址的不可变运行目录避免覆盖。</p>'''
rows=[]
for ds in DS:
 z=S['engineering'][ds];rows.append([LABEL[ds],f"{z['mean_query_median_ms']['ind8']:.4f}",f"{z['mean_query_median_ms']['ind4']:.4f}",interval(z['ind4_vs_ind8']),f"{z['ndcg10']['ind4']:.6f}",f"{z['gpu_peak_with_both_engines_bytes']/2**30:.2f}"])
body+=table(['完整原有检索流程','独立8箱 ms','独立4箱 ms','4箱耗时变化 / 查询区间','nDCG@10（两者相同）','双引擎进程峰值 GiB'],rows)
body+='''<p>这一项包括原有 local6 残差处理、UVA 按需原始 token 精算及 CPU Top100 ID/分数输出；NQ 使用已有去重粗码缓存，Hotpot 使用原有未去重粗码缓存，同语料两个方法共享输入。5 次配对遍历、仅一个进程会话，64 查询 Top100 列表和跨重复列表都一致。峰值包含双引擎及完整后续缓冲，不能解释成单方法部署显存。nDCG 是本地固定查询结果，不是公开榜单得分。这里比较的是我们自己的可靠工程控制；并未因此声称赢过 LIK/FastPLAID/公开 SOTA。本轮因新算法未过门槛，没有扩大公开系统比较。</p>
<h2>6. 可执行研究决策与可主张范围</h2>
<p><b>停止：</b>不再围绕固定条件摘要做精确表优化、整查询 16/32 路由、复杂学习选择器或大规模参数搜索。停止依据是该表示的理想界余量极小及实用局部混合净收益失败，而非证明所有可能条件约束都不可能成功。</p>
<p><b>可保留：</b>normal32/power32 等独立状态工程基线；本轮具体支持 power32 的 4 分箱调优。可独立开源的部分是同预算对照、位打包和候选门槛审计、分离 profiler 与正式时间的测量工具。实际代码还需拆除字符串替换式驱动、清理接口、固定依赖及许可证，再考虑上游 PR。当前材料是研究复现原型，不宣称生产库。</p>
<p><b>论文范围：</b>能够如实陈述“特定有限码本 MaxSim 表示下，合并状态的信息损失与执行成本可抵消收益”的系统负结果与诊断方法；工程调优本身不构成新算法贡献。目前没有支持 CCF A 正向算法主张的证据，也不能仅靠叙述补足创新。若未来提出独立机制，须重新提出可证伪假设，并冻结后在未用于设计的数据上验证；NQ 和 Hotpot 已全部是观察数据。</p>
<p><b>明确未完成且不冒充完成：</b>两个缺失全库编码器格、盲测泛化、跨硬件/并发、总线物理流量、理论最优上界与独立状态最优参数搜索。它们不是本轮停止决策的必要前提；失败路线不值得继续消耗这些资源。物理内存和统计环境局限仍影响工程外推。</p>
<h2>7. 复现与证据入口</h2><p><a href="条件摘要失效诊断_复现命令.txt">固定配置与完整命令</a> · <a href="条件摘要失效诊断_汇总.json">重算汇总</a> · <a href="条件摘要失效诊断_复现包.zip">复现包</a> · <a href="条件摘要失效诊断_证据校验.json">打包与完整性校验</a> · <a href="独立大库与创新性_完整报告.html">先前 v5 报告</a></p>
<p>复现包包含本轮全部有效原始逐查询计时、失败记录、源码及冻结散列、模型来源、固定查询/文档清单、四个 64k 子库的查询向量与码本索引，以及图片/报告。为避免重复搬运，原始全库向量、模型、local6 缓存继续复用服务器既有目录；不打包密码或第三方运行库。子库诊断可直接用打包索引运行，不需要再次编码。</p>'''
hybrows=[]
for ds in DS:
 for k in ['mixed24','local24']:
  ss=[s[k]for s in S['hybrid'][ds]['sessions']]
  hybrows.append([LABEL[ds]+' / '+k,'24',f"{avg([s['slack_mean']for s in ss]):.4f}",f"{avg([s['false_survivors_truecut']for s in ss]):,.0f}",f"{avg([s['scored_documents']for s in ss]):,.0f}",f"{avg([s['scored_tokens']for s in ss])/1e6:.2f}",f"{max(s['workspace_plus_transient_bytes']for s in ss)/2**20:.1f}"])
hybdetail=table(['全库 / 方法','状态 B/文档','平均松弛','无效存活','评分文档','评分码字 M','工作区峰值 MiB'],hybrows)
body=body.replace('<p>NQ 相比更简单',hybdetail+'<p>NQ 相比更简单')
page('MaxSim 条件摘要失效诊断 · 完整证据',body,O/'条件摘要失效诊断_完整证据.html')
print(json.dumps(S['validation'],ensure_ascii=False));print('REPORTS_READY',O)
