"""Run an explicit prediction-only job plan across independent GPUs inside tmux."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from .core import file_hash, load_benchmark, write_json


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    root=Path(plan['workspace'])
    output=Path(plan['output_root'])
    info,_=load_benchmark(plan['benchmark'])
    if info['benchmark_id']!=plan['benchmark_id']:
        raise ValueError('Benchmark changed since plan creation')
    jobs=Queue()
    for job in plan['jobs']:
        jobs.put(job)
    state={j['name']:{'state':'pending','output':j['output'],'batch_size':j['batch_size']} for j in plan['jobs']}
    lock=threading.Lock()
    status_path=output/'status.json'

    def update(name,**fields):
        with lock:
            if 'progress_checked_at' in fields and state[name]['state'] != 'running':
                return
            state[name].update(fields)
            write_json(status_path,{'updated_at':now(),'tmux_session':plan['tmux_session'],
                                   'benchmark_id':plan['benchmark_id'],'jobs':state})

    def worker(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
                 PYTHONUNBUFFERED='1',USE_TF='0',TF_CPP_MIN_LOG_LEVEL='3')
        while True:
            try:
                job=jobs.get_nowait()
            except Empty:
                return
            name=job['name']
            log=output/'logs'/f'{name}.log'
            started=time.monotonic()
            try:
                for path,expected in plan['source_sha256'].items():
                    if file_hash(root/path)!=expected:
                        raise ValueError(f'Inference code changed after queue creation: {path}')
                update(name,state='running',gpu=gpu,started_at=now(),log=str(log))
                print(f"[{now()}] GPU {gpu} START {name} bs={job['batch_size']}",flush=True)
                with log.open('a') as stream:
                    stream.write(f'\nQUEUE START {now()} GPU={gpu}\n')
                    stream.flush()
                    proc=subprocess.Popen(job['command'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    update(name,pid=proc.pid)
                    code=proc.wait()
                    if code:
                        raise RuntimeError(f'Prediction process exited {code}; see {log}')
                    # Format/completeness validation only; do not calculate pair scores.
                    subprocess.run([sys.executable,'-m','eval_astra_bench','validate',
                                    '--benchmark',plan['benchmark'],'--predictions',job['output']],
                                   cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                update(name,state='complete',finished_at=now(),elapsed_seconds=round(time.monotonic()-started,2),
                       trajectories=info['counts']['trajectories'],frames=info['counts']['frames'])
                print(f'[{now()}] GPU {gpu} COMPLETE {name}',flush=True)
            except Exception as exc:
                update(name,state='failed',finished_at=now(),error=str(exc),
                       elapsed_seconds=round(time.monotonic()-started,2))
                print(f'[{now()}] GPU {gpu} FAILED {name}: {exc}',flush=True)
            finally:
                jobs.task_done()

    def heartbeat():
        with lock:
            snapshot=json.loads(json.dumps(state))
        for name,job in snapshot.items():
            if job['state']=='running':
                p=Path(job['output'])
                count=0
                if p.exists():
                    with p.open('rb') as f:
                        count=sum(1 for line in f if line.endswith(b'\n'))
                update(name,trajectories=count,progress_checked_at=now())
        with lock:
            counts=Counter(job['state'] for job in state.values())
            active=', '.join(f"gpu{j.get('gpu')}:{name} {j.get('trajectories',0)}/{info['counts']['trajectories']}"
                             for name,j in state.items() if j['state']=='running')
        print(f'[{now()}] {dict(counts)} | {active}',flush=True)

    (output/'logs').mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(plan['gpus'])) as pool:
        futures=[pool.submit(worker,gpu) for gpu in plan['gpus']]
        while any(not f.done() for f in futures):
            time.sleep(30)
            heartbeat()
        for f in futures:
            f.result()
    heartbeat()
    failed=[name for name,job in state.items() if job['state']!='complete']
    print(f'[{now()}] QUEUE FINISHED: {len(state)-len(failed)}/{len(state)} complete; failed={failed}',flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
