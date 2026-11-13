import torch
import torch.nn.functional as F
from transformers import DynamicCache

def fork_cache(cache,batch=1):
 return DynamicCache(ddp_cache_data=[(layer.keys.expand(batch,-1,-1,-1),layer.values.expand(batch,-1,-1,-1)) for layer in cache.layers])

@torch.inference_mode()
def cached_scores(model,cache,next_logits,completions,batch_size=16):
 length=cache.get_seq_length();device=next_logits.device
 widths={len(c) for c in completions};assert len(widths)==1
 first_logp=F.log_softmax(next_logits.float(),dim=-1)
 scores=[]
 for start in range(0,len(completions),batch_size):
  ids=torch.tensor(completions[start:start+batch_size],device=device)
  fork=fork_cache(cache,len(ids))
  result=model(input_ids=ids[:,:-1],past_key_values=fork,use_cache=True,logits_to_keep=ids.shape[1]-1)
  rest=F.log_softmax(result.logits.float(),dim=-1).gather(-1,ids[:,1:,None]).squeeze(-1).sum(-1)
  total=(first_logp[0,ids[:,0]]+rest)/ids.shape[1]
  scores.extend(total.cpu().tolist())
  del result,fork,ids,rest,total
 assert cache.get_seq_length()==length
 return scores

@torch.inference_mode()
def uncached_scores(model,prompt,completions):
 output=[]
 for c in completions:
  ids=torch.tensor([prompt+c],device=model.device)
  logits=model(input_ids=ids,use_cache=False,logits_to_keep=len(c)+1).logits[:,:-1].float()
  values=F.log_softmax(logits,dim=-1).gather(-1,torch.tensor(c,device=model.device)[None,:,None])
  output.append(float(values.mean().cpu()))
 return output
