import sys
import jax
import jax.numpy as jnp
import time
import torch
import torch_xla2

from torch.utils import _pytree as pytree

def t2j(torchtensor):
    device = jax.devices('gpu')[0]
    torchtensor = torchtensor.detach().cpu()
    jax_array = torch_xla2.tensor.t2j(torchtensor)
    return jax.device_put(jax_array, device)

def benchmark_torch(model, sample_inputs, compile=False):
    if compile:
        model = torch.compile(model)
    
    for i in range(3):
        start = time.perf_counter()
        x = model(*sample_inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        yield end - start

def benchmark_torchxla2(model, sample_inputs, dynamo=False):
    import torch_xla2.export

    exported = torch.export.export(model, sample_inputs)
    weights, jax_func = torch_xla2.export.exported_program_to_jax(exported)
    jax_func = jax.jit(jax_func)

    sample_inputs = pytree.tree_map_only(torch.Tensor, torch_xla2.tensor.t2j, sample_inputs)

    for i in range(3):
        start = time.perf_counter()
        res = jax_func(weights, sample_inputs)
        # jax need to wait
        for r in res:
            r.block_until_ready()
        end = time.perf_counter()
        yield end - start




import torchbenchmark
import torchbenchmark.models.maml_omniglot
models = [
'torchbenchmark.models.dcgan',
'torchbenchmark.models.LearningToPaint',
'torchbenchmark.models.hf_T5_base',
#'torchbenchmark.models.maml_omniglot',
#'torchbenchmark.models.shufflenet_v2_x1_0',
#'torchbenchmark.models.Background_Matting'
]
import importlib

def main():
    print('Running for: ', sys.argv)
    name = sys.argv[1]
    type_ = sys.argv[2]
    m = importlib.import_module('torchbenchmark.models.' + name)
    model, sample_inputs = m.Model(test="eval", device="cuda").get_module()
    success = True
    times = [0, 0, 0]
    try:
        if type_ == 'torch_eager':
            times= list(benchmark_torch(model, sample_inputs))
        elif type_ == 'torch_inductor':
            times = list(benchmark_torch(model, sample_inputs, True))
        elif type_ == 'xla2':
            times= list(benchmark_torchxla2(model, sample_inputs, True))
    except:
        import traceback
        traceback.print_exc()
        success = False

    with open('times.csv', 'a') as f:
        print(
            ','.join(
                map(str, [name, type_, 'success' if success else 'fail'] + times)
            ), file=f)


if __name__ == '__main__':
    main()