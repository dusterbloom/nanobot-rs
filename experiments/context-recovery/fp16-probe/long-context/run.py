from pathlib import Path
import os, subprocess, json, hashlib, time
root=Path('/Users/peppi/Dev/nanobot-rs/experiments/context-recovery/endurance-fp16-long')
root.mkdir(exist_ok=False)
binary=Path('/private/tmp/higgs-fp16-artifacts/bench_frontier-fp16-attn')
command=['/usr/bin/time','-l',str(binary),'--model-dir','/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2','--frontiers','4096,8192,16384','--probe-tokens','128','--runs','1','--prefill-chunk-size','1024','--format','json']
provenance={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'command':command,'order':['off1','on1','on2','off2'],'results':[]}
(root/'provenance.json').write_text(json.dumps(provenance,indent=2))
for label in provenance['order']:
    flag='1' if label.startswith('on') else '0'
    env=dict(os.environ,HIGGS_ESCHA_NATIVE='1',HIGGS_FP16_KV_PROBE=flag,HIGGS_FP16_ATTN_PROBE=flag)
    started=time.time()
    print('START',label,flush=True)
    with (root/(label+'.json')).open('w') as out,(root/(label+'.log')).open('w') as log:
        try:
            result=subprocess.run(command,env=env,stdout=out,stderr=log,timeout=900)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    provenance['results'].append({'label':label,'exit':code,'seconds':time.time()-started})
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('DONE',provenance['results'][-1],flush=True)
    if code:
        raise SystemExit(code)
(root/'COMPLETE').write_text('Four fresh-process frontier sweeps completed; compare JSON digests and timings.\n')
