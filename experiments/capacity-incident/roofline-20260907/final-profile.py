import pathlib
source=pathlib.Path('/private/tmp/higgs-roofline-evidence/final-validation.py').read_text();exec(source[:source.index('assert "\'AC Power\'"')])
assert "'AC Power'" in subprocess.check_output(['pmset','-g','batt'],text=True)
stop_existing();server=None
try:
 env=os.environ.copy()
 for key in ['HIGGS_ESCHA_TRELLIS_GEMM','HIGGS_CHUNKED_PREFILL_CHUNK_SIZE','HIGGS_ESCHA_FUSED_OUTPUT_HAD']:env.pop(key,None)
 env.update(HIGGS_ENABLE_THINKING='1',HIGGS_PROFILE='1')
 server=launch('final-profile16k',env)
 run_measure(server,'final-profile16k',16000,'profile16k')
finally:
 if server is not None and server.poll() is None:
  server.send_signal(signal.SIGINT)
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 subprocess.run(['tmux','new-session','-d','-c','/private/tmp','-s','higgs-roofline-live','cd /Users/peppi/Dev/higgs && HIGGS_ENABLE_THINKING=1 /Users/peppi/.local/bin/higgs serve --mlx-profile throughput > /private/tmp/higgs-roofline-live.log 2>&1'],check=True)
