set -e
# for name in $(cat models); do
#   for type in "xla_lazy_tensor" "xla_dynamo" ; do
#     PJRT_DEVICE=CUDA LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hanq/miniconda3/envs/py310_2/lib/ python run_bench_torchxla2.py $name $type
#   done
# done

for name in $(cat models); do
  for type in "xla2" "torch_eager" "torch_inductor" ; do
    python run_bench_torchxla2.py $name $type
  done
done