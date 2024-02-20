set -e
for name in $(cat models); do
  for type in "xla2" "torch_eager" "torch_inductor"; do
    python run_bench_torchxla2.py $name $type
  done
done