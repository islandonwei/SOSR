"""Two explicit single-factor training ablations; no dataset or inference changes."""
import hashlib,json,random
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

def split_pools(packed,positive):
    row=[int(x) for x in packed if x!=-1]
    if row.count(-2)!=1 or row.count(-3)!=1:raise ValueError('Malformed hierarchy separators')
    a,b=row.index(-2),row.index(-3)
    if a>=b:raise ValueError('Hierarchy pool order invalid')
    pools=[row[:a],row[a+1:b],row[b+1:]];flat=[j for pool in pools for j in pool]
    if any(j<0 or j==positive for j in flat) or len(set(flat))!=len(flat):raise ValueError('Invalid, duplicate, overlapping or positive negative IDs')
    return pools

class NoContrastive:
    def __init__(self):self.calls=0;self.rows=0
    def __call__(self,queries,categories,positives,packed,temperature):
        assert not categories.requires_grad,'Original detached category cache must be retained'
        self.calls+=1;self.rows+=len(queries)
        return queries.sum()*0.0

class HierarchyRandom:
    def __init__(self,seed=42,quotas=(10,5,5)):
        self.seed=seed;self.quotas=quotas;self.rng=random.Random(seed);self.calls=0;self.rows=0
        self.budget_histogram=Counter();self.positive_trace=[];self.negative_trace=[];self.group_trace=[];self.batch_sizes=[]
    def rng_sha256(self):return hashlib.sha256(json.dumps(self.rng.getstate(),separators=(',',':')).encode()).hexdigest()
    def select(self,pools):
        groups=[self.rng.sample(pool,min(k,len(pool))) for pool,k in zip(pools,self.quotas)]
        return sorted(j for group in groups for j in group),[len(group) for group in groups]
    def __call__(self,queries,categories,positives,packed,temperature):
        assert temperature==.05
        assert not categories.requires_grad,'Original detached category cache must be retained'
        self.calls+=1
        logits=queries@categories.T/temperature
        rows=packed.detach().cpu().tolist();pos=positives.detach().cpu().tolist();losses=[];chosen=[];group_counts=[]
        for i,positive in enumerate(pos):
            pools=split_pools(rows[i],positive)
            selected,counts=self.select(pools)
            assert len(selected)==len(set(selected)) and all(0<=j<len(categories) and j!=positive for j in selected)
            self.rows+=1;self.budget_histogram[len(selected)]+=1;chosen.append(selected+[-1]*(20-len(selected)));group_counts.append(counts)
            if not selected:continue
            indices=torch.tensor(selected,dtype=torch.long,device=queries.device)
            values=torch.cat((logits[i,positive].reshape(1),logits[i,indices]))
            losses.append(-F.log_softmax(values,dim=0)[0])
        self.positive_trace.append(np.array(pos,dtype=np.int16));self.negative_trace.append(np.array(chosen,dtype=np.int16))
        self.group_trace.append(np.array(group_counts,dtype=np.int8));self.batch_sizes.append(len(pos))
        return torch.stack(losses).mean() if losses else queries.sum()*0.0
    def save_trace(self,path):
        np.savez_compressed(path,positive_indices=np.concatenate(self.positive_trace),selected_negative_indices=np.concatenate(self.negative_trace),
                            group_counts=np.concatenate(self.group_trace),batch_sizes=np.array(self.batch_sizes,dtype=np.int16))
