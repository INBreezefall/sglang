set -euo pipefail
cd /bench
site=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
sdk=/usr/local/Ascend/cann-9.1.0
abi=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
if [ ! -d kernel-947 ]; then
  cp -a base/kernel kernel-947.staging
  (cd kernel-947.staging && git apply /bench/pr-947.patch)
  mv kernel-947.staging kernel-947
fi
for variant in main pr; do
  if [ "$variant" = main ]; then src=base/kernel; else src=kernel-947; fi
  style=${REGISTRATION_STYLE:-native}
  if [[ "$style" = paired ]]; then opname=evidence_$variant; else opname=npu; fi
  dest=/bench/op-$variant/sgl_kernel_npu
  if [[ "$style" = paired ]]; then dest=/bench/paired-$variant; fi
  mkdir -p "$dest/lib"
  cp "$src/python/sgl_kernel_npu/sgl_kernel_npu/"{__init__.py,kvcacheio.py} "$dest/"
  cat > /bench/register.cpp <<'CPP'
#include "torch_helper.h"
#include "sgl_kenel_npu_ops.h"
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("transfer_kv_dim_exchange(Tensor device_k, Tensor host_k, Tensor device_v, Tensor host_v, Tensor device_indices, Tensor host_indices, int page_size, int direct, int flags) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("transfer_kv_dim_exchange", TORCH_FN(sglang::npu_kernel::transfer_kv_dim_exchange));
}
CPP
  if [[ "$style" = paired ]]; then sed -i "s/FRAGMENT(npu/FRAGMENT($opname/; s/IMPL(npu/IMPL($opname/" /bench/register.cpp; fi
  inc=(-I"$src/csrc/utils" -I"$src/include" "-I$site/torch/include" "-I$site/torch/include/torch/csrc/api/include" "-I$site/torch_npu/include" "-I$site/torch_npu/include/third_party/hccl" "-I$sdk/include" "-I$sdk/aarch64-linux/include" "-I$sdk/include/experiment/metadef" "-I$sdk/include/experiment/platform" "-I$sdk/include/experiment/runtime" -I/usr/local/python3.12.13/include/python3.12)
  c++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI="$abi" "${inc[@]}" /bench/register.cpp "$src/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp" -L"$site/torch/lib" -L"$site/torch_npu/lib" -L"$sdk/lib64" -L"$sdk/aarch64-linux/lib64" -Wl,-Bsymbolic-functions,-rpath,"$site/torch/lib:$site/torch_npu/lib:$sdk/lib64" -ltorch_npu -ltorch_cpu -lc10 -lascendcl -o "$dest/lib/libsgl_kernel_npu.so"
  sha256sum "$dest/lib/libsgl_kernel_npu.so" "$src/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp"
done
