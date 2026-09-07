import argparse,json,pathlib,time,tomllib,urllib.request,threading,sys
from tokenizers import Tokenizer
sys.path.insert(0,'/Users/peppi/Dev/nanobot-rs/experiments/context-recovery')
from memory_probe import memory_sample
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--tokens',type=int,default=16000);p.add_argument('--pid',type=int,default=75074);p.add_argument('--label',default='baseline');p.add_argument('--output',type=int,default=256);p.add_argument('--prompt-file');p.add_argument('--request-file');p.add_argument('--session-id',type=int);a=p.parse_args()
r=pathlib.Path(a.out);r.mkdir(parents=True,exist_ok=False)
c=tomllib.loads(pathlib.Path('/Users/peppi/.config/higgs/config.toml').read_text());h={'Authorization':'Bearer '+c['server']['api_key'],'Content-Type':'application/json'}
def req(path,body=None,timeout=10):return urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:9000'+path,data=None if body is None else json.dumps(body).encode(),headers=h),timeout=timeout)
x=json.load(req('/metrics'))['capacity'];assert x['activeReservations']==0 and x['queuedWaiters']==0,x
T=Tokenizer.from_file('/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/tokenizer.json')
head='Performance archive '+a.label+'. Read the archive before following the final instruction.\n'
line='Archive record: maintenance and inventory were reviewed. No policy or ownership changes were recorded.\n'
tail='\nFinal instruction: output consecutive integers from 1 through 1000, one integer per line. Start now with 1. Do not explain.\n'
lo,hi=0,a.tokens
while lo+1<hi:
 m=(lo+hi)//2
 if len(T.encode(head+line*m+tail).ids)<=a.tokens:lo=m
 else:hi=m
prompt=pathlib.Path(a.prompt_file).read_text() if a.prompt_file else head+line*lo+tail;(r/'prompt.txt').write_text(prompt)
payload={'model':'escha-35b-a3b','messages':[{'role':'user','content':prompt}],'max_tokens':a.output,'temperature':0,'stream':True,'return_progress':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
if a.session_id is not None:payload['session_id']=a.session_id
if a.request_file:
 payload=json.loads(pathlib.Path(a.request_file).read_text());(r/'prompt.txt').write_text('\n\n'.join('['+m['role']+']\n'+m['content'] for m in payload['messages']))
(r/'request.json').write_text(json.dumps(payload));stop=threading.Event()
def monitor():
 with (r/'memory.jsonl').open('w') as f:
  while not stop.is_set():
   try:f.write(json.dumps({'at':time.time(),'memory':memory_sample(a.pid),'capacity':json.load(req('/metrics'))['capacity']})+'\n');f.flush()
   except Exception as e:f.write(json.dumps({'error':str(e)})+'\n');f.flush()
   stop.wait(2)
t=threading.Thread(target=monitor);t.start();start=time.monotonic();first=None;last=None;usage=None;finish=None;parts=[];prog=[]
try:
 with req('/v1/chat/completions',payload,600) as response,(r/'stream.jsonl').open('w') as f:
  for raw in response:
   if not raw.startswith(b'data:'):continue
   b=raw[5:].strip()
   if b==b'[DONE]':break
   e=json.loads(b);elapsed=time.monotonic()-start;f.write(json.dumps({'seconds':elapsed,'event':e})+'\n');f.flush()
   if e.get('error'):raise RuntimeError(str(e['error']))
   if e.get('prompt_progress'):prog.append(e['prompt_progress'])
   if e.get('usage'):usage=e['usage']
   for choice in e.get('choices',[]):
    d=choice.get('delta',{});s=d.get('content') or d.get('reasoning_content')
    if s:
     if first is None:first=elapsed
     last=elapsed;parts.append(s)
    if choice.get('finish_reason'):finish=choice['finish_reason']
 out={'label':a.label,'seconds':time.monotonic()-start,'ttft_seconds':first,'last_token_seconds':last,'usage':usage,'finish':finish,'progress':prog,'answer':''.join(parts),'native_user_tokens':None if a.request_file else len(T.encode(prompt).ids)}
 assert usage and finish in ['stop','length'],out
 (r/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['answer','progress']}),flush=True)
finally:stop.set();t.join()
