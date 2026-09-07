from pathlib import Path
import subprocess,json,time,re
s=Path('/private/tmp/higgs-roofline-evidence/final-validation.py').read_text();exec(s[:s.index('assert "\'AC Power\'"')])
p=Path('/private/tmp/higgs-fused-profile');power=subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]
(p/'conditions-causal.json').write_text(json.dumps({'power':power,'battery':subprocess.check_output(['pmset','-g','batt'],text=True),'thermal':subprocess.check_output(['pmset','-g','therm'],text=True)},indent=2))
stop_existing()
try:
 subprocess.run(['python3','build.py'],cwd=p,check=True)
 for arm,name in [(1,'query128'),(5,'original-causal'),(4,'reg-explicit'),(7,'reg-causal')]:
  before=vm().get('Swapouts',0)
  with (p/(name+'.log')).open('w') as log:
   env=os.environ.copy();env['MTL_CAPTURE_ENABLED']='1'
   proc=subprocess.Popen([str(p/'probe'),'1024','32768',str(arm),str(p/(name+'.gputrace'))],stdout=log,stderr=subprocess.STDOUT,cwd=p,env=env)
   try:
    while proc.poll() is None:
     time.sleep(1)
     assert vm().get('Swapouts',0)==before,'New swapouts'
     assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1,'Memory pressure'
   finally:
    if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
  print(name,proc.returncode,flush=True)
  assert proc.returncode==0,(p/(name+'.log')).read_text()
 print('COMPLETE',flush=True)
finally:
 time.sleep(20)
 subprocess.run(['tmux','new-session','-d','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
 for _ in range(90):
  try:print('RESTORED',json.dumps(metrics()['capacity']),flush=True);break
  except (OSError,ValueError):time.sleep(1)
 else:raise RuntimeError('Restored server did not become ready')
