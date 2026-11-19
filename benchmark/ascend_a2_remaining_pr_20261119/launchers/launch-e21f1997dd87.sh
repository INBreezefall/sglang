set -euo pipefail
role=$1
rank=$2
arm=$3
node=$4
case_name=$5
cd /bench
out=/bench/services/$case_name
mkdir -p "$out" /bench/pd-cann-cache /bench/pd-triton /bench/pd-sgl-cache "$out/prom"
exec >"$out/server.log" 2>&1
set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
export PYTHONPATH="/bench/src/$arm/python:/bench/wheel:/bench/deps:${PYTHONPATH:-}"
export PROMETHEUS_MULTIPROC_DIR="$out/prom"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_CACHE_DIR=/bench/pd-sgl-cache TRITON_CACHE_DIR=/bench/pd-triton ASCEND_CACHE_PATH=/bench/pd-cann-cache
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TASK_QUEUE_ENABLE=1
export GLOO_SOCKET_IFNAME=bond0 HCCL_SOCKET_IFNAME=bond0 TP_SOCKET_IFNAME=bond0
export HCCL_IF_IP="${NODE_IP}" SGLANG_HOST_IP="${NODE_IP}"
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=180 HCCL_BUFFSIZE=256
export HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0
export HCCL_IF_BASE_PORT=29580 HCCL_NPU_SOCKET_PORT_RANGE=46100-46999
export ASCEND_MF_STORE_URL=tcp://${PREFILL_HEAD}:18883 ASCEND_MF_TRANSFER_PROTOCOL=device_rdma
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python SGLANG_HICACHE_HOST_MEM_BACKEND=acl
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=180 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=180
policy=total_tokens
args=()
case "$arm" in
  48096) export SGLANG_PYTHON_RADIX_EVICT_TAIL_CHUNK_SIZE=256 ;;
  48520) export SGLANG_PD_DECODE_CACHE_INPUT_ONLY=1 ;;
  48537) export SGLANG_MOONCAKE_ASCEND_REGISTER_ALLOCATION=1 ;;
  48540) export SGLANG_NPU_PREFILL_MOE_TILE_TOKENS=256 ;;
  48543) export SGLANG_NPU_HICACHE_HOST_ALLOCATOR=aclrt ;;
  48574) export SGLANG_NPU_DEEPEP_NORMAL_BF16=1 ;;
  48575) export SGLANG_NPU_PREWARM_PINNED_HOST=1 ;;
  48576) export SGLANG_HICACHE_PREFETCH_TIMING_TRACE=1 SGLANG_HICACHE_TRACE_RUN_ID=measured ;;
  48630) export SGLANG_DP_ROUND_TRACE=1 ;;
  49183) if test "$role" = prefill; then args+=(--hicache-draft-size 0.12); fi ;;
esac
dtype=bf16
transport=ascend
ep=1
a2a=none
capacity=32768
context=16384
profile=${6:-standard}
if [[ "$profile" == pinned ]]; then unset SGLANG_HICACHE_HOST_MEM_BACKEND; export PYTORCH_NPU_ALLOC_CONF=pinned_reserve_segment_size_mb:64; fi
if [[ "$profile" == eviction && "$role" == decode ]]; then capacity=8192; context=8192; fi
if [[ "$profile" == storage && "$role" == prefill ]]; then
 capacity=8192; context=8192
 export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="/bench/services/$case_name/storage" SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=16GB
 mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
 args+=(--hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete)
fi
if [[ "$profile" == deep ]]; then
 ep=16; a2a=deepep
 export HCCL_BUFFSIZE=512 DEEPEP_HCCL_BUFFSIZE=1536
 args+=(--deepep-mode auto --deepep-dispatcher-output-dtype int8)
 export DEEP_NORMAL_MODE_USE_INT8_QUANT=0
 vendor=/usr/local/python3.12.13/lib/python3.12/site-packages/deep_ep/vendors/hwcomputing
 export LD_LIBRARY_PATH="$vendor/op_api/lib:$LD_LIBRARY_PATH" ASCEND_CUSTOM_OPP_PATH="$vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
 export LD_PRELOAD="$vendor/op_api/lib/libcust_opapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [[ "$profile" == *mooncake* ]]; then
 transport=mooncake
 mcdir=/bench/mooncake-npu
 if [[ "$profile" == spec-mooncake* ]]; then mcdir=/bench/mooncake-014; fi
 export ASCEND_BENCH_MOONCAKE_SUPPORT=thread
 if [[ "$profile" == spec-mooncake* || "$profile" == *normalized* ]]; then export ASCEND_BENCH_MOONCAKE_SUPPORT=thread-regions; fi
 if [[ "$profile" == *-stock ]]; then
  unset ASCEND_BENCH_MOONCAKE_SUPPORT
  export PYTHONPATH="$mcdir:$PYTHONPATH"
 else
  export PYTHONPATH="/bench/moon-fixture:$mcdir:$PYTHONPATH"
 fi
 export ASCEND_BUFFER_POOL=0:0
 export ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=1 SGLANG_MOONCAKE_ASCEND_USE_DEVICE_SUFFIX=0
fi
if [[ "$profile" == spec* ]]; then
 args+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-quantization modelslim)
fi
if [[ "$role" == prefill ]]; then
  dist="${PREFILL_HEAD}:${7}"
  args+=(--enable-hierarchical-cache --hicache-ratio 1.5 --hicache-size 0 --hicache-write-policy write_through --hicache-io-backend kernel_ascend --hicache-mem-layout page_first_kv_split --schedule-policy lpm)
else
  dist="${DECODE_HEAD}:${7}"
  args+=(--disaggregation-decode-enable-radix-cache)
fi
cmd=(python3 -u -m sglang.launch_server --model-path /model --served-model-name glm52 --trust-remote-code --device npu --dtype bfloat16 --attention-backend ascend --sampling-backend ascend --quantization modelslim --tp-size 16 --attn-dp-size 8 --ep-size "$ep" --moe-a2a-backend "$a2a" --nnodes 2 --node-rank "$rank" --dist-init-addr "$dist" --dist-timeout 360 --page-size 128 --kv-cache-dtype "$dtype" --context-length "$context" --max-total-tokens "$capacity" --chunked-prefill-size 4096 --max-prefill-tokens 8192 --max-running-requests 32 --mem-fraction-static 0.9 --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled --disable-shared-experts-fusion --host "${NODE_IP}" --port 18684 --enable-metrics --enable-cache-report --disaggregation-mode "$role" --disaggregation-transfer-backend "$transport" --disaggregation-bootstrap-port 18784 --load-balance-method "$policy" --skip-server-warmup "${args[@]}")
printf '%s\n' "${cmd[@]}" >"$out/args.txt"
printf '%s\n' "$profile" >"$out/profile"
cat "/bench/src/$arm/SHA" >"$out/source-sha"
python3 -c 'import torch, torch_npu, sgl_kernel_npu, transformers; print("runtime",torch.__version__,torch_npu.__version__,transformers.__version__,sgl_kernel_npu.__file__,flush=True)'
set +e
"${cmd[@]}"
rc=$?
printf '%s\n' "$rc" > "$out/exit"
exit "$rc"
