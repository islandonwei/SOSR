"""Nearest-rank quantiles for fixed-budget right-censored query runtimes."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics

BUDGET=30000.0
METHODS=['FSSP','ROSE','DROMC','LayeredDijkstra']
LABELS={'FSSP':'FSSP (certified geometric bound)',
        'ROSE':'ROSE (RNII-guided reimplementation)',
        'DROMC':'DROMC (existing Candidate-A* implementation)',
        'LayeredDijkstra':'Layered multi-source Dijkstra'}

def quantile(values,p):
    if not values:raise ValueError('empty sample')
    assert 0<p<=1
    ordered=sorted(math.inf if v is None else v for v in values)
    v=ordered[math.ceil(p*len(values))-1]
    return None if math.isinf(v) else v

def format_ms(x):return '>30000' if x is None else f'{x:.6f}'

def summarize(rows,queries,require_complete=True):
    expected={q['id']:q for q in queries};per_query=defaultdict(dict)
    for r in rows:
        assert r['id'] in expected and r['algorithm'] in METHODS
        assert r['repeat'] in [1,2,3] and r['status'] in ['ok','timeout']
        key=(r['id'],r['algorithm'])
        assert r['repeat'] not in per_query[key],('duplicate measurement',key,r['repeat'])
        if r['status']=='ok':
            assert isinstance(r['planner_ms'],(float,int)) and 0<=r['planner_ms']<=BUDGET
        else:assert r['planner_ms'] is None and r['censor_ms']==BUDGET
        per_query[key][r['repeat']]=r
    if require_complete:
        assert len(per_query)==len(expected)*4
        assert all(set(x)=={1,2,3} for x in per_query.values())
    groups=defaultdict(list)
    for (qid,method),reps in per_query.items():
        if len(reps)!=3:continue
        q=expected[qid];key=(q['dataset'],q['k'],q['frequency'],method)
        groups[key].append((q,[reps[i] for i in [1,2,3]]))
    summary=[]
    for (dataset,k,freq,method),items in sorted(groups.items()):
        n=len(items);aggregate=[];repeat_stats=[]
        for repeat in [1,2,3]:
            xs=[rr[repeat-1]['planner_ms'] for q,rr in items]
            repeat_stats.append({'repeat':repeat,'n':n,'median_ms':quantile(xs,.5),'p95_ms':quantile(xs,.95),
                                 'timeouts':sum(x is None for x in xs),'timeout_rate':sum(x is None for x in xs)/n})
        aggregate=[quantile([r['planner_ms'] for r in rr],.5) for q,rr in items]
        counts=[c for q,rr in items for c in q['candidate_counts']]
        r={'dataset':dataset,'length':k,'frequency':freq,'algorithm':method,'display_name':LABELS[method],
           'n_unique_queries':n,'measurements':3*n,'median_ms':quantile(aggregate,.5),'p95_ms':quantile(aggregate,.95),
           'timeout_runs':sum(r['status']=='timeout' for q,rr in items for r in rr),
           'timeout_rate_runs':sum(r['status']=='timeout' for q,rr in items for r in rr)/(3*n),
           'queries_with_any_timeout':sum(any(r['status']=='timeout' for r in rr) for q,rr in items),
           'queries_with_censored_median':sum(x is None for x in aggregate),
           'candidate_count_min':min(counts),'candidate_count_median':statistics.median(counts),'candidate_count_max':max(counts),
           'per_layer_counts':[{'min':min(q['candidate_counts'][i] for q,rr in items),
                                'median':statistics.median(q['candidate_counts'][i] for q,rr in items),
                                'max':max(q['candidate_counts'][i] for q,rr in items)} for i in range(k)],
           'repeat_statistics':repeat_stats,'counter_scope':'completed runs only; counts for timed-out runs are unknown'}
        metrics=['settled_vertices','scanned_edges'] if method=='LayeredDijkstra' else ['h2h_calls']
        if method=='ROSE':metrics+=['pair_evaluations','refinements']
        r['counters']={}
        for metric in metrics:
            xs=[x[metric] for q,rr in items for x in rr if x['status']=='ok']
            r['counters'][metric]={'completed_runs':len(xs),'median':statistics.median(xs) if xs else None}
        summary.append(r)
    return summary

def write_report(root,rows,queries):
    result=summarize(rows,queries)
    doc={'complete':True,'unique_queries':len(queries),'measurement_records':len(rows),'timeout_budget_ms':BUDGET,
         'primary_quantiles':'Nearest-rank quantiles over per-query median of three paired repetitions. Censored values remain unknown above budget; null quantile means >30000ms.',
         'timeout_rate':'Timed-out runs / all runs, with any-timeout query count also reported.',
         'timing_scope':'Distance plus ordered service-vertex reconstruction; no full road-walk recovery or HSM.',
         'DROMC_identity':'Existing Candidate-A* implementation retained at explicit user request; not a verified reproduction of the cited full DROMC algorithm.',
         'results':result}
    (root/'RESULTS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['dataset','length','frequency','algorithm','display_name','n_unique_queries','median_ms','p95_ms','timeout_runs','timeout_rate_runs','candidate_count_min','candidate_count_median','candidate_count_max']
    with (root/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result:
            small={k:row[k] for k in fields};small['median_ms']=format_ms(row['median_ms']);small['p95_ms']=format_ms(row['p95_ms']);writer.writerow(small)
    lines=['Four-method routing comparison','DROMC denotes the explicitly retained existing Candidate-A* implementation; full-paper identity is not claimed.',
           'Primary latency returns distance + service vertices; excludes HSM and full road-walk recovery.',
           'Milliseconds. Quantiles use all queries with censoring; timeout rate uses all three repetitions.','']
    for row in result:
        lines.append(f"{row['dataset']} k={row['length']} {row['frequency']} {row['algorithm']}: n={row['n_unique_queries']}, median={format_ms(row['median_ms'])}, P95={format_ms(row['p95_ms'])}, timeout={row['timeout_rate_runs']:.2%}")
    (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return doc

def selftest():
    assert quantile(list(range(1,93))+[None]*8,.95) is None
    assert quantile(list(range(1,96))+[None]*5,.95)==95
    assert quantile([1,2,None],.5)==2
    assert quantile([1,None,None],.5) is None
    assert quantile([None]*50,.5) is None
    assert quantile(list(range(1,51)),.95)==48
    queries=[{'id':'q','dataset':'NYC','k':1,'frequency':'L','candidate_counts':[3]}]
    rows=[]
    for m in METHODS:
        for rep in [1,2,3]:
            rows.append({'id':'q','algorithm':m,'repeat':rep,'status':'ok','planner_ms':rep,
                         'h2h_calls':10,'settled_vertices':20,'scanned_edges':30,'pair_evaluations':40,'refinements':2})
    assert all(r['median_ms']==2 for r in summarize(rows,queries))
    try:summarize(rows+[rows[0]],queries)
    except AssertionError:pass
    else:raise AssertionError('duplicate accepted')
    try:summarize(rows[:-1],queries)
    except AssertionError:pass
    else:raise AssertionError('missing repetition accepted')
    print(json.dumps({'passed':True,'tests':'censor propagation, percentile threshold, repeated-query aggregation, duplicate and incomplete rejection'}))

if __name__=='__main__':selftest()
