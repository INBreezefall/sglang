set -euo pipefail
rank=$1
master=$2
interface_ip=$3
port=$4
arm=$5
flag=$6
round=$7
set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
cd "/bench/jobs/${8:-ep16-cross}"
export ARM_SHA=$(cat "/bench/src/$arm/SHA")
export PYTHONPATH="/bench/src/$arm/python:/bench/wheel:/bench/deps:${PYTHONPATH:-}"
export DEEPEP_OUT="$PWD/round-$round" ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TASK_QUEUE_ENABLE=1 HF_HUB_OFFLINE=1
export ASCEND_CACHE_PATH="$DEEPEP_OUT/cann" TRITON_CACHE_DIR="$DEEPEP_OUT/triton" SGLANG_CACHE_DIR="$DEEPEP_OUT/sglang"
export GLOO_SOCKET_IFNAME=bond0 HCCL_SOCKET_IFNAME=bond0 HCCL_IF_IP="$interface_ip"
export HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=180
export HCCL_BUFFSIZE=512 DEEPEP_GROUP_BUFFER_MB=3072 DEEPEP_HCCL_BUFFSIZE=3072
export DEEP_NORMAL_MODE_USE_INT8_QUANT=0 SGLANG_NPU_DEEPEP_NORMAL_BF16="$flag"
vendor=/usr/local/python3.12.13/lib/python3.12/site-packages/deep_ep/vendors/hwcomputing
export LD_LIBRARY_PATH="$vendor/op_api/lib:$LD_LIBRARY_PATH" ASCEND_CUSTOM_OPP_PATH="$vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_PRELOAD="$vendor/op_api/lib/libcust_opapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
mkdir -p "$ASCEND_CACHE_PATH" "$TRITON_CACHE_DIR" "$SGLANG_CACHE_DIR"
python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank="$rank" --master-addr="$master" --master-port="$port" probe.py 48574 "$flag" "$DEEPEP_OUT/group.json"
