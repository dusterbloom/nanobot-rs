import pathlib
source=pathlib.Path('/private/tmp/higgs-roofline-evidence/final-validation.py').read_text()
source=source[:source.index('assert "\'AC Power\'"')]
power_guard="   if \"'AC Power'\" not in subprocess.check_output(['pmset','-g','batt'],text=True):raise RuntimeError('Power source changed; timing invalid')"
assert power_guard in source
source=source.replace(power_guard, "   with (root/(out+'.power.jsonl')).open('a') as power_log:power_log.write(json.dumps({'at':time.time(),'power':subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True)})+'\\n')")
exec(source)
(root/'glm-profile32k-conditions.json').write_text(json.dumps({'policy':'User authorized profiling regardless of power; do not compare timing directly to AC baseline','power_before':subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True),'settings':subprocess.check_output(['/usr/bin/pmset','-g','custom'],text=True)},indent=2))
stop_existing();server=None
try:
 env=os.environ.copy()
 for key in ['HIGGS_ESCHA_TRELLIS_GEMM','HIGGS_CHUNKED_PREFILL_CHUNK_SIZE','HIGGS_ESCHA_FUSED_OUTPUT_HAD']:env.pop(key,None)
 env.update(HIGGS_ENABLE_THINKING='1',HIGGS_PROFILE='1')
 server=launch('glm-profile32k',env)
 run_measure(server,'glm-profile32k',32000,'profile32k')
finally:
 if server is not None and server.poll() is None:
  server.send_signal(signal.SIGINT)
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 subprocess.run(['tmux','new-session','-d','-c','/private/tmp','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
