import json,pathlib,statistics
root=pathlib.Path('/private/tmp/higgs-dense-prefill/serving');out=[]
for p in sorted(root.glob('32k-*/result.json')):
 if 'warmup' in str(p):continue
 r=json.loads(p.read_text());samples=[json.loads(s) for s in (p.parent/'memory.jsonl').read_text().splitlines()];valid=[s for s in samples if s.get('memory')]
 out.append({'name':p.parent.name,'seconds':r['seconds'],'ttft':r['ttft_seconds'],'decode_interval':r['last_token_seconds']-r['ttft_seconds'],'usage':r['usage'],'answer':r.get('answer'),'peakGiB':max(s['memory']['physical_footprint_bytes'] for s in valid)/2**30,'swapouts':valid[-1]['memory']['system_vm_counters']['Swapouts']-valid[0]['memory']['system_vm_counters']['Swapouts'],'pressure':sorted(set(s.get('capacity',{}).get('pressure','missing') for s in samples))})
summary={'runs':out}
if len(out)==3:
 controls=[out[0],out[2]];candidate=out[1]
 for key in ['seconds','ttft','decode_interval','peakGiB']:
  base=statistics.mean(r[key] for r in controls);summary[key]={'control_mean':base,'candidate':candidate[key],'change_percent':100*(candidate[key]/base-1)}
 summary['answers_identical']=len(set(r['answer'] for r in out))==1
 summary['control_ttft_drift_percent']=100*(out[2]['ttft']/out[0]['ttft']-1)
(root/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
