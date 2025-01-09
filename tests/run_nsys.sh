nsys profile \
  --trace cuda,osrt,nvtx \
#  --gpu-metrics-device=all \
  --cuda-memory-usage true \
  --force-overwrite true \
  --output profile_run_v1 \
