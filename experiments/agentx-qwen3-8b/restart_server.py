import json,os,pathlib,signal,subprocess,sys,time,urllib.request
p=pathlib.Path('/mnt/sda1/hwx/Agentrix/experiments/agentx-qwen3-8b')
label=sys.argv[1]
pidfile=p/'server.pid'
if pidfile.exists():
 pid=int(pidfile.read_text())
 try:
  cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'vllm' not in cmd or b'18000' not in cmd: raise RuntimeError('PID does not match this benchmark server')
  os.kill(pid,signal.SIGTERM)
 except FileNotFoundError: pass
 for _ in range(60):
  if not pathlib.Path(f'/proc/{pid}').exists(): break
  time.sleep(1)
 else: raise RuntimeError('previous server did not stop')
log=p/'server.log'
if log.exists():log.rename(p/f'server-before-{label}-{int(time.time())}.log')
with log.open('w') as out:
 proc=subprocess.Popen(['bash',str(p/'serve.sh')],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
 if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
 try:
  urllib.request.urlopen('http://127.0.0.1:18000/health',timeout=2)
  break
 except Exception:time.sleep(2)
else: raise RuntimeError('server readiness timeout')
body={'model':'Qwen3-8B','messages':[{'role':'user','content':'Reply briefly.'}],'max_tokens':4,'ignore_eos':True}
r=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=90))
assert r['usage'].get('prompt_tokens_details') is not None,r['usage']
print('server ready',proc.pid,r['usage'],flush=True)
