import json,pathlib,subprocess,tomllib,urllib.request,time,re
p=pathlib.Path('/private/tmp/higgs-dense-prefill')
c=tomllib.loads(pathlib.Path('/Users/peppi/.config/higgs/config.toml').read_text())
def capacity():
 req=urllib.request.Request('http://127.0.0.1:9000/metrics',headers={'Authorization':'Bearer '+c['server']['api_key']})
 return json.load(urllib.request.urlopen(req,timeout=3))['capacity']
def swaps():
 return int(re.search(r'Swapouts:\s+(\d+)',subprocess.check_output(['vm_stat'],text=True))[1])
def idle():
 level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
 assert level==1, f'OS memory pressure level {level}; stop benchmark'
print(subprocess.check_output(['pmset','-g','batt'],text=True),flush=True)
for q,k in [(1024,8192),(1024,32768),(1024,45056)]:
 idle();before=swaps();print('SHAPE',q,k,flush=True)
 with (p/f'tiled-{q}-{k}.jsonl').open('w') as log:
  proc=subprocess.Popen([str(p/'query-tiled'),str(q),str(k)],stdout=log,stderr=subprocess.STDOUT,cwd=p)
  try:
   while proc.poll() is None:
    time.sleep(1);idle();assert swaps()==before,'New swapping; stop benchmark'
  except BaseException:
   proc.terminate();proc.wait();raise
 print((p/f'tiled-{q}-{k}.jsonl').read_text(),flush=True)
 assert proc.returncode==0,proc.returncode
print('COMPLETE',flush=True)
