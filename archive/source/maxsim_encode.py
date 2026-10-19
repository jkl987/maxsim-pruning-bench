"""Minimal ColBERT encoder following the pinned upstream implementation.
No retriever training or remote model code. All model parameters strictly loaded.
"""
import argparse
import csv
import hashlib
import json
import string
import zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).parent

class Encoder:
    def __init__(self):
        path = ROOT/'maxsim-assets/model'
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.meta = json.loads((path/'artifact.metadata').read_text())
        state = load_file(str(path/'model.safetensors'))
        config = BertConfig.from_pretrained(path, local_files_only=True)
        config._attn_implementation = 'sdpa'
        self.bert = BertModel(config)
        self.bert.load_state_dict({k[5:]:v for k,v in state.items() if k.startswith('bert.')}, strict=True)
        assert set(state) == {'linear.weight'} | {'bert.'+k for k in self.bert.state_dict()}
        self.bert.eval().to('cuda',dtype=torch.float32)
        self.linear = state['linear.weight'].cuda().float()
        self.skip = {self.tok.encode(s,add_special_tokens=False)[0] for s in string.punctuation}
        self.skip.add(self.tok.pad_token_id)
        print('model loaded',len(state),self.linear.shape,flush=True)

    @torch.inference_mode()
    def encode(self, texts, query=False):
        result = []
        for start in range(0,len(texts),32):
            batch = texts[start:start+32]
            maxlen = self.meta['query_maxlen'] if query else self.meta['doc_maxlen']
            obj = self.tok(batch,padding='max_length' if query else 'longest',truncation=True,
                           max_length=maxlen-1,return_tensors='pt')
            marker = self.tok.convert_tokens_to_ids(self.meta['query_token_id' if query else 'doc_token_id'])
            ids = obj['input_ids']
            ids = torch.cat([ids[:,:1],torch.full((len(batch),1),marker),ids[:,1:]],1)
            mask = obj['attention_mask']
            mask = torch.cat([mask[:,:1],torch.ones((len(batch),1),dtype=mask.dtype),mask[:,1:]],1)
            if query:
                ids[ids==self.tok.pad_token_id] = self.tok.mask_token_id
                if self.meta['attend_to_mask_tokens']:mask[:] = 1
                keep = ids != self.tok.pad_token_id
            else:
                keep = torch.tensor([[x not in self.skip for x in row] for row in ids.tolist()])
            h = self.bert(ids.cuda(),attention_mask=mask.cuda()).last_hidden_state
            v = F.normalize(F.linear(h,self.linear),dim=-1).cpu()
            result.extend([row[k].contiguous() for row,k in zip(v,keep)])
            if start%512==0:print('encode',query,start,len(texts),flush=True)
        return result

def qrels(path):
    result = {}
    if not path.exists(): return result
    with path.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            result.setdefault(row['query-id'],{})[row['corpus-id']]=int(row['score'])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);args=parser.parse_args()
    name=args.dataset
    data=ROOT/'maxsim-data';data.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/(name+'.zip')) as z:
        for item in z.infolist():
            dest=(data/item.filename).resolve()
            assert dest.is_relative_to(data.resolve())
        z.extractall(data)
    source=data/name
    docs=sorted([json.loads(s) for s in (source/'corpus.jsonl').read_text().splitlines()],key=lambda d:d['_id'])
    queries={d['_id']:d['text'] for d in map(json.loads,(source/'queries.jsonl').read_text().splitlines())}
    train=qrels(source/'qrels/train.tsv'); test=qrels(source/'qrels/test.tsv')
    # Fixed disjoint training query subset, selected before any scoring.
    train_ids=sorted(set(train)-set(test),key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:32]
    assert len(train_ids)==32
    query_ids=train_ids+sorted(test)
    enc=Encoder()
    # Published model-card sanity example, unrelated to evaluation corpora.
    sq=enc.encode(['Which planet is known as the Red Planet?'],query=True)[0]
    sd=enc.encode(["Venus is often called Earth's twin because of its similar size and proximity.",
                  'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
                  'Jupiter, the largest planet in our solar system, has a prominent red spot.',
                  'Saturn, famous for its rings, is sometimes mistaken for the Red Planet.'])
    scores=[float((sq@d.T).max(-1).values.sum()) for d in sd]
    assert scores[1]==max(scores),scores
    dvec=enc.encode([(d.get('title','')+' '+d['text']).strip() for d in docs])
    qvec=enc.encode([queries[k] for k in query_ids],query=True)
    out=ROOT/('maxsim-'+name);out.mkdir(exist_ok=True)
    torch.save(dict(doc_ids=[d['_id'] for d in docs],query_ids=query_ids,train_count=32,
                    docs=dvec,queries=qvec,test_qrels=test,train_qrels=train),out/'vectors.pt')
    provenance=dict(model='answerdotai/answerai-colbert-small-v1',revision='934fa8bb4ce2284f4c2baa232d81aca4d076fa5e',
        encoding_reference='https://github.com/stanford-futuredata/ColBERT/tree/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c',
        implementation='Minimal reimplementation of upstream tensorize/query/doc, strict full state dict, FP32 SDPA',
        query_maxlen=enc.meta['query_maxlen'],doc_maxlen=enc.meta['doc_maxlen'],sanity_scores=scores,
        doc_count=len(dvec),train_queries=train_ids,test_query_count=len(test),
        vector_sha256=hashlib.sha256((out/'vectors.pt').read_bytes()).hexdigest())
    (out/'encoding.json').write_text(json.dumps(provenance,indent=2));(out/'encoded').write_text('done\n')
    print(provenance,flush=True)

if __name__=='__main__':main()
