import pathlib
source=pathlib.Path('/private/tmp/higgs-roofline-evidence/final-validation.py').read_text();source=source[:source.index('assert "\'AC Power\'"')]
source=source.replace("str(root/'final-candidate'/'higgs')","'/private/tmp/higgs-dense-prefill/candidate/higgs'")
source=source.replace("   if \"'AC Power'\" not in subprocess.check_output(['pmset','-g','batt'],text=True):raise RuntimeError('Power source changed; timing invalid')", "   power_now=subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]\n   if power_now!=power_source:raise RuntimeError('Power source changed; timing invalid')")
exec(source)
root=pathlib.Path('/private/tmp/higgs-dense-prefill/serving');root.mkdir(exist_ok=True)
# Keep the proven measurement utility path while writing into this experiment.
measure_source=pathlib.Path('/private/tmp/higgs-roofline-evidence/measure.py')
import shutil,hashlib
shutil.copy2(measure_source,root/'measure.py')
power_source=subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]
(root/'conditions.json').write_text(json.dumps({'power':subprocess.check_output(['pmset','-g','batt'],text=True),'settings':subprocess.check_output(['pmset','-g','custom'],text=True),'sha256':hashlib.sha256(pathlib.Path('/private/tmp/higgs-dense-prefill/candidate/higgs').read_bytes()).hexdigest()},indent=2))
stop_existing();server=None
try:
 for index,enabled in enumerate([0,1,0]):
  tag=f'32k-{index}-block{enabled}'
  env=os.environ.copy()
  for key in ['HIGGS_PROFILE','HIGGS_CHUNKED_PREFILL_CHUNK_SIZE','HIGGS_ESCHA_TRELLIS_GEMM']:env.pop(key,None)
  env.update(HIGGS_ENABLE_THINKING='1',HIGGS_DENSE_PREFILL_BLOCK=str(enabled))
  server=launch(tag,env)
  try:
   run_measure(server,tag+'-warmup',512,'warmup')
   run_measure(server,tag,32000,'dense-prefill32k')
   r=json.loads((root/tag/'result.json').read_text());print(json.dumps({'tag':tag,'result':r}),flush=True)
  finally:
   if server.poll() is None:server.send_signal(signal.SIGINT)
   try:server.wait(timeout=20)
   except subprocess.TimeoutExpired:server.kill();server.wait()
   server=None
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=20)
 subprocess.run(['tmux','new-session','-d','-c','/private/tmp','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
print('COMPLETE',flush=True)
