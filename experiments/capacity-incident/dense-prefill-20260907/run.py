import json,pathlib,subprocess,tomllib,urllib.request,time,re
p=pathlib.Path('/private/tmp/higgs-dense-prefill')
c=tomllib.loads(pathlib.Path('/Users/peppi/.config/higgs/config.toml').read_text())
def capacity():
 req=urllib.request.Request('http://127.0.0.1:9000/metrics',headers={'Authorization':'Bearer '+c['server']['api_key']})
 return json.load(urllib.request.urlopen(req,timeout=3))['capacity']
def swaps():
 return int(re.search(r'Swapouts:\s+(\d+)',subprocess.check_output(['vm_stat'],text=True))[1])
def idle():
 m=capacity();assert m['activeReservations']==0 and m['queuedWaiters']==0 and m['pressure']=='normal','Server busy or memory pressure; stop benchmark'
print(subprocess.check_output(['pmset','-g','batt'],text=True),flush=True)
for q,k in [(32,1024),(31,1055),(1024,1024),(1024,8192),(1024,32768),(1024,45056)]:
 idle();before=swaps();print('SHAPE',q,k,flush=True)
 with (p/f'probe-{q}-{k}.jsonl').open('w') as log:
  proc=subprocess.Popen([str(p/'probe'),str(q),str(k)],stdout=log,stderr=subprocess.STDOUT,cwd=p)
  try:
   while proc.poll() is None:
    time.sleep(1);idle();assert swaps()==before,'New swapping; stop benchmark'
  except BaseException:
   proc.terminate();proc.wait();raise
 print((p/f'probe-{q}-{k}.jsonl').read_text(),flush=True)
 assert proc.returncode==0,proc.returncode
print('COMPLETE',flush=True)
