import os,sys,time,json,pathlib,subprocess,signal,tomllib,urllib.request
root=pathlib.Path('/private/tmp/higgs-roofline-evidence')
c=tomllib.loads(pathlib.Path('/Users/peppi/.config/higgs/config.toml').read_text())
headers={'Authorization':'Bearer '+c['server']['api_key']}
def metrics():
 return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:9000/metrics',headers=headers),timeout=3))
def stop_existing():
 m=metrics()['capacity'];assert m['activeReservations']==0 and m['queuedWaiters']==0,'Server busy; preserving user request'
 ps=subprocess.run(['lsof','-t','-iTCP:9000','-sTCP:LISTEN'],capture_output=True,text=True).stdout.split();assert len(ps)==1,ps
 pid=int(ps[0]);cmd=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True);assert '/higgs serve' in cmd,cmd
 os.kill(pid,signal.SIGINT)
 for _ in range(100):
  try:os.kill(pid,0)
  except ProcessLookupError:return
  time.sleep(.2)
 raise RuntimeError('Server did not stop')
def vm():
 import re
 s=subprocess.check_output(['vm_stat'],text=True);return {k.strip():int(v) for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',s,re.M)}
def run_measure(server,out,tokens,label,request_file=None):
 initial=vm();args=[sys.executable,str(root/'measure.py'),str(root/out),'--tokens',str(tokens),'--output','256' if tokens==45000 else '128','--pid',str(server.pid),'--label',label]
 if tokens==45000:args+=['--prompt-file','/private/tmp/higgs-recovery-45k-final-result/prompt.txt','--session-id','2026090701']
 if request_file:args+=['--request-file',str(request_file)]
 p=subprocess.Popen(args)
 try:
  while p.poll() is None:
   time.sleep(1)
   delta=vm().get('Swapouts',0)-initial.get('Swapouts',0)
   if delta>512:raise RuntimeError(f'New swapping during measurement: {delta} pages')
   if server.poll() is not None:raise RuntimeError('Server exited')
   if "'AC Power'" not in subprocess.check_output(['pmset','-g','batt'],text=True):raise RuntimeError('Power source changed; timing invalid')
  if p.returncode:raise RuntimeError(f'Measurement exit {p.returncode}')
 finally:
  if p.poll() is None:p.terminate();p.wait(timeout=10)
def launch(tag,env):
 for attempt in range(3):
  path=root/(tag+f'.startup{attempt}.log')
  before=vm()
  with path.open('w') as log:
   server=subprocess.Popen([str(root/'final-candidate'/'higgs'),'serve','--mlx-profile','throughput'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd='/Users/peppi/Dev/higgs')
  ready=False
  for _ in range(90):
   if server.poll() is not None:break
   try:
    metrics();ready=True;break
   except (OSError,ValueError):time.sleep(1)
  (root/(tag+f'.startup{attempt}.vm.json')).write_text(json.dumps({'before':before,'after':vm(),'ready':ready}))
  if ready:return server
  if server.poll() is None:server.terminate();server.wait(timeout=20)
  if 'critical memory pressure' not in path.read_text() or attempt==2:
   raise RuntimeError(f'Server startup failed; see {path}')
  print(f'{tag}: transient critical startup rejection; retrying after 30 seconds',flush=True)
  time.sleep(30)
 raise RuntimeError('unreachable')
assert "'AC Power'" in subprocess.check_output(['pmset','-g','batt'],text=True), 'AC required'
stop_existing()
try:
 for index,enabled in enumerate([1]):
  tag='final-default45k'
  env=os.environ.copy()
  for key in ['HIGGS_ESCHA_TRELLIS_GEMM','HIGGS_CHUNKED_PREFILL_CHUNK_SIZE','HIGGS_PROFILE','HIGGS_ESCHA_FUSED_OUTPUT_HAD']:env.pop(key,None)
  env['HIGGS_ENABLE_THINKING']='1'
  server=launch(tag,env)
  try:
   run_measure(server,tag+'-warmup',512,'warmup')
   run_measure(server,tag,45000,'matched45k')
   result=json.loads((root/tag/'result.json').read_text())
   assert all(v in result['answer'] for v in ['LANTERN-4729-QZ','COPPER-8163-VX','Neri']),result['answer']
   payload=json.loads((root/tag/'request.json').read_text())
   payload['messages'] += [{'role':'assistant','content':result['answer']},{'role':'user','content':"Return only the current owner's name."}]
   payload['max_tokens']=32
   request_file=root/(tag+'-continuation-request.json');request_file.write_text(json.dumps(payload))
   run_measure(server,tag+'-continuation',0,'continuation',request_file)
   result=json.loads((root/(tag+'-continuation')/'result.json').read_text())
   assert result['answer'].strip().strip('.')=='Neri',result['answer']
   assert any(p.get('cache',0)>44000 for p in result['progress']), 'No retained prefix reuse: '+str(result['progress'])

  finally:
   if server.poll() is None:server.send_signal(signal.SIGINT)
   try:server.wait(timeout=20)
   except subprocess.TimeoutExpired:server.kill();server.wait()
finally:
 subprocess.run(['tmux','new-session','-d','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
