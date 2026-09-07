from pathlib import Path
import subprocess,json,time,re
s=Path('/private/tmp/higgs-roofline-evidence/final-validation.py').read_text();exec(s[:s.index('assert "\'AC Power\'"')])
p=Path('/private/tmp/higgs-causal-peel');power=subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]
(p/'conditions-causal.json').write_text(json.dumps({'power':power,'battery':subprocess.check_output(['pmset','-g','batt'],text=True),'thermal':subprocess.check_output(['pmset','-g','therm'],text=True)},indent=2))
stop_existing()
try:
 subprocess.run(['python3','build.py'],cwd=p,check=True)
 print('BUILD PASSED',flush=True)
 for q,k in [(31,1055),(1024,32768)]:
  before=vm().get('Swapouts',0);print('SHAPE',q,k,flush=True)
  with (p/f'causal-{q}-{k}.jsonl').open('w') as log:
   proc=subprocess.Popen([str(p/'probe'),str(q),str(k)],stdout=log,stderr=subprocess.STDOUT,cwd=p)
   try:
    while proc.poll() is None:
     time.sleep(1)
     assert subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]==power,'Power changed'
     assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1,'Memory pressure'
     assert vm().get('Swapouts',0)==before,'New swapouts'
   finally:
    if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
  print('SHAPE EXIT',q,k,proc.returncode,flush=True)
  assert proc.returncode==0,'Numerical/build/runtime gate failed; inspect shape log'
 print('COMPLETE',flush=True)
finally:
 time.sleep(20)
 subprocess.run(['tmux','new-session','-d','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
 for _ in range(90):
  try:print('RESTORED',json.dumps(metrics()['capacity']),flush=True);break
  except (OSError,ValueError):time.sleep(1)
 else:raise RuntimeError('Restored server did not become ready')
