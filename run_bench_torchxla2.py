import sys
import time
import torch

from torch.utils import _pytree as pytree

def input_like(sample_inputs, device='cpu'):
    res = []
    for p in sample_inputs:
        if isinstance(p, torch.Tensor):
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                res.append(torch.rand(p.shape, dtype=p.dtype).to(device))
            else:
                res.append(p.clone().to(device))
        else:
            res.append(p)
    return res

def benchmark_torch(model, sample_inputs, compile=False):
    model = model.to('cuda')
    if compile:
        model = torch.compile(model)
    
    for i in range(3):
        inputs = input_like(sample_inputs, device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = model(*inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        yield end - start


def benchmark_torchxla(model, sample_inputs, dynamo=False):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

    model = model.to(device)
    if dynamo:
        model = torch.compile(model, backend='openxla')
    
    for i in range(3):
        inputs = input_like(sample_inputs, device)
        xm.mark_step(wait=True) # make sure inputs are on device already
        start = time.perf_counter()
        res = model(*inputs)
        xm.mark_step(wait=True)
        end = time.perf_counter()
        yield end - start


def benchmark_torchxla2(model, sample_inputs, dynamo=False):
    import torch_xla2.export
    import torch_xla2
    import jax
    import jax.numpy as jnp

    device = jax.devices()[0]

    exported = torch.export.export(model, sample_inputs)
    weights, jax_func = torch_xla2.export.exported_program_to_jax(exported)
    jax_func = jax.jit(jax_func)
    weights = pytree.tree_map_only(jnp.ndarray, lambda x : jax.device_put(x, device), weights)


    for i in range(3):
        inputs = input_like(sample_inputs, 'cuda')
        inputs = pytree.tree_map_only(torch.Tensor, lambda x : jax.device_put(torch_xla2.tensor.t2j(x), device), inputs)

        for i in inputs:
            i.block_until_ready()
        start = time.perf_counter()
        res = jax_func(weights, inputs)
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
    times = [0, 0, 0]
    success = False
    try:
        m = importlib.import_module('torchbenchmark.models.' + name)
        model, sample_inputs = m.Model(test="eval", device="cpu").get_module()
        if type_ == 'torch_eager':
            times= list(benchmark_torch(model, sample_inputs))
        elif type_ == 'torch_inductor':
            times = list(benchmark_torch(model, sample_inputs, True))
        elif type_ == 'xla2':
            times= list(benchmark_torchxla2(model, sample_inputs, True))
        if type_ == 'xla_lazy_tensor':
            times = list(benchmark_torchxla(model, sample_inputs, False))
        elif type_ == 'xla_dynamo':
            times = list(benchmark_torchxla(model, sample_inputs, True))
        success = True
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
