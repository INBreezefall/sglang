# Start one genuine loopback SeaweedFS S3 instance for this prefill host and export its client configuration.
store_run=$1
store_runtime=${OBJECT_RUNTIME:-/bench/object-runtime}
mkdir -p "$store_run/storage"
export PYTHONPATH="$store_runtime/python:${PYTHONPATH:-}"
store_base=$(python3 - <<'PY'
import socket
for port in (17731,17931,17131):
 sockets=[]
 try:
  for p in range(port,port+10):
   s=socket.socket();sockets.append(s);s.bind(('127.0.0.1',p))
  print(port);break
 except OSError:pass
 finally:
  for sock in sockets:sock.close()
else:raise RuntimeError('no free port block for the local test storage')
PY
)
store_s3=$((store_base+3))
printf '%s\n' "$store_s3" >"$store_run/object-port"
(umask 077; python3 -c 'import secrets;print(secrets.token_hex(24))' >"$store_run/.object-secret")
export AWS_ACCESS_KEY_ID=sglang-local-test
AWS_SECRET_ACCESS_KEY=$(cat "$store_run/.object-secret");export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1 AWS_EC2_METADATA_DISABLED=true
GOMAXPROCS=4 "$store_runtime/weed" mini -dir="$store_run/storage" -ip=127.0.0.1 -ip.bind=127.0.0.1 \
  -master.port="$store_base" -volume.port="$((store_base+1))" -filer.port="$((store_base+2))" -s3.port="$store_s3" -admin.port="$((store_base+4))" \
  -master.port.grpc="$((store_base+5))" -volume.port.grpc="$((store_base+6))" -filer.port.grpc="$((store_base+7))" -s3.port.grpc="$((store_base+8))" -admin.port.grpc="$((store_base+9))" \
  -admin.port.legacy=0 -webdav=false -s3.port.iceberg=0 -s3.port.lance=0 -master.volumeSizeLimitMB=256 -volume.max=128 -bucket=sglang-hicache >"$store_run/object-service.log" 2>&1 &
store_pid=$!
python3 - "$store_s3" "$store_pid" <<'PY'
import os,sys,time
import boto3
from botocore.config import Config
port,pid=map(int,sys.argv[1:]);client=boto3.client('s3',endpoint_url=f'http://127.0.0.1:{port}',config=Config(s3={'addressing_style':'path'},connect_timeout=1,read_timeout=3,retries={'max_attempts':0}))
until=time.monotonic()+50
while time.monotonic()<until:
 try:
  os.kill(pid,0);client.head_bucket(Bucket='sglang-hicache')
  payload=b'Ascend SGLang local S3 readiness'
  client.put_object(Bucket='sglang-hicache',Key='health/object',Body=payload)
  assert client.get_object(Bucket='sglang-hicache',Key='health/object')['Body'].read()==payload
  client.delete_object(Bucket='sglang-hicache',Key='health/object')
  print('SeaweedFS S3 put/get/delete passed',flush=True);break
 except ProcessLookupError:raise RuntimeError('local storage process exited')
 except Exception as e:
  why=type(e).__name__;time.sleep(.5)
else:raise RuntimeError('local S3 readiness deadline '+why)
PY
export SGLANG_BENCH_OBJECT_ENDPOINT="http://127.0.0.1:$store_s3"
