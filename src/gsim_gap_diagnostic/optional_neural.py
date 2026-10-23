"""Independent local LSTM/TCN fits accelerated along a batch-of-models axis.

Every job owns all parameter and optimizer-state slices. No learned weight,
target, gradient norm, stopping rule, or normalization is shared between jobs.
Only visible target months contribute to each independently normalized loss.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import sys
sys.dont_write_bytecode = True
from dataclasses import dataclass, asdict
import hashlib
import time
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (20261014, 20261015, 20261016)
MODEL_NAMES = ('lstm', 'tcn')


@dataclass(frozen=True)
class NeuralConfig:
    hidden_size: int = 16
    epochs: int = 100
    learning_rate: float = .01
    weight_decay: float = 1e-4
    gradient_clip: float = 5.
    sequence_length: int = 12
    input_features: int = 28

    def validate(self):
        if self.hidden_size not in (16, 32) or self.epochs not in (50, 100, 200):
            raise ValueError('Use the prespecified small capacity/epoch grid.')
        if self.sequence_length != 12 or self.input_features != 28:
            raise ValueError('The common visible representation is fixed at 12 x 28.')
        if self.learning_rate != .01 or self.weight_decay != 1e-4 or self.gradient_clip != 5.:
            raise ValueError('Optimizer constants must remain fixed for this round.')


def configure_runtime():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _weights(seeds, shape, fan_in, fan_out):
    # A job's initialization depends on its stable fit identity and seed, never batching.
    bound = np.sqrt(6. / (fan_in + fan_out))
    values = [np.random.default_rng(seed).uniform(-bound, bound, shape).astype(np.float32)
              for seed in seeds]
    return nn.Parameter(torch.from_numpy(np.stack(values)))


def _derived(seeds, suffix):
    return [int.from_bytes(hashlib.sha256(f'{int(seed)}|{suffix}'.encode()).digest()[:8], 'little')
            for seed in seeds]


class IndependentLSTM(nn.Module):
    def __init__(self, seeds, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.wx = _weights(_derived(seeds, 'lstm_wx'), (features, 4*hidden), features, 4*hidden)
        self.wh = _weights(_derived(seeds, 'lstm_wh'), (hidden, 4*hidden), hidden, 4*hidden)
        bias = torch.zeros((len(seeds), 1, 4*hidden)); bias[:, :, hidden:2*hidden] = 1.
        self.bias = nn.Parameter(bias)
        self.head = _weights(_derived(seeds, 'lstm_head'), (hidden, 1), hidden, 1)
        self.intercept = nn.Parameter(torch.zeros((len(seeds), 1, 1)))

    def forward(self, x):
        batch, samples, steps, features = x.shape
        projected = torch.bmm(x.reshape(batch, samples*steps, features), self.wx)
        projected = projected.reshape(batch, samples, steps, 4*self.hidden)
        h = torch.zeros((batch, samples, self.hidden), device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for step in range(steps):
            i, f, g, o = (projected[:, :, step] + torch.bmm(h, self.wh) + self.bias).chunk(4, dim=-1)
            c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
            h = torch.sigmoid(o)*torch.tanh(c)
        return (torch.bmm(h, self.head) + self.intercept).squeeze(-1)


class IndependentTCN(nn.Module):
    """Three causal convolution layers, kernel 3, dilations 1/2/4 (15-month RF)."""
    def __init__(self, seeds, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.kernels = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer, inputs in enumerate((features, hidden, hidden)):
            self.kernels.append(_weights(_derived(seeds, f'tcn_kernel{layer}'),
                                         (3, inputs, hidden), 3*inputs, 3*hidden))
            self.biases.append(nn.Parameter(torch.zeros((len(seeds), 1, 1, hidden))))
        self.head = _weights(_derived(seeds, 'tcn_head'), (hidden, 1), hidden, 1)
        self.intercept = nn.Parameter(torch.zeros((len(seeds), 1, 1)))

    def forward(self, x):
        batch, samples, steps, _ = x.shape
        for layer, dilation in enumerate((1, 2, 4)):
            result = self.biases[layer].expand(batch, samples, steps, self.hidden)
            for tap in range(3):
                lag = tap*dilation
                shifted = x if not lag else F.pad(x[:, :, :-lag, :], (0, 0, lag, 0))
                result = result + torch.bmm(shifted.reshape(batch, samples*steps, x.shape[-1]),
                                            self.kernels[layer][:, tap]).reshape(batch, samples, steps, self.hidden)
            x = torch.relu(result)
        return (torch.bmm(x[:, :, -1, :], self.head) + self.intercept).squeeze(-1)


def _per_job_clip(parameters, max_norm):
    # Global clip_grad_norm_ would couple jobs: explicitly avoid it.
    norm_squared = None
    for parameter in parameters:
        value = parameter.grad.square().flatten(1).sum(1)
        norm_squared = value if norm_squared is None else norm_squared + value
    scale = torch.clamp(max_norm/(torch.sqrt(norm_squared)+1e-12), max=1.)
    for parameter in parameters:
        parameter.grad.mul_(scale.reshape((-1,)+(1,)*(parameter.ndim-1)))


def validate_prepared(prepared, config):
    x = np.asarray(prepared['X_sequence'], dtype=np.float32)
    y = np.asarray(prepared['y_standardized'], dtype=np.float32)
    observed = np.asarray(prepared['observed_mask'], dtype=bool)
    query = np.asarray(prepared['query_indices'], dtype=int)
    if x.ndim != 3 or x.shape[1:] != (config.sequence_length, config.input_features):
        raise ValueError('Expected the common [calendar, 12, 28] input.')
    if y.shape != (len(x),) or observed.shape != y.shape or not np.isfinite(x).all():
        raise ValueError('Invalid input shape or nonfinite explanatory values.')
    if len(query) == 0 or np.any(query < 0) or np.any(query >= len(x)):
        raise ValueError('Query indices missing or outside the calendar.')
    if np.any(observed[query]) or not np.isnan(y[query]).all():
        raise ValueError('Current query target values must already be hidden.')
    if not np.array_equal(np.isfinite(y), observed):
        raise ValueError('Supervision must contain exactly the visible target values.')
    if not observed.any():
        raise ValueError('No observed supervision; use the declared common fallback.')
    if {'truth', 'target_truth', 'query_truth', 'R', 'NSE', 'pass_joint'}.intersection(prepared):
        raise ValueError('Accuracy labels or target truth cannot enter a neural fit.')
    return x, y, observed


def fit_many_jobs(jobs: Sequence[dict], model_name: str, config: NeuralConfig | dict | None = None,
                  device: str = 'cuda') -> list[dict]:
    """jobs=[{'prepared': visible_dict, 'seed': int, 'job_id': str}, ...].

    All jobs must have the same calendar length. Caller handles declared
    fallbacks before batching. Predictions cover the original calendar, so the
    caller can apply the identical visible-residual bridge to every algorithm.
    """
    if not jobs:
        return []
    config = NeuralConfig(**config) if isinstance(config, dict) else config or NeuralConfig()
    config.validate()
    if model_name not in MODEL_NAMES:
        raise ValueError('Only the registered LSTM and TCN are supported.')
    configure_runtime()
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; do not silently change runtime.')
    seeds = [int(job['seed']) for job in jobs]
    if not set(seeds).issubset(SEEDS):
        raise ValueError('Use the three frozen round seeds.')
    identities = [job.get('job_id', job['prepared'].get('case_id')) for job in jobs]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError('Every independent local fit needs a stable case/fit identity.')
    initialization_seeds = [int.from_bytes(hashlib.sha256(
        f'neural-local-fit-v1|{identity}|{seed}'.encode()).digest()[:8], 'little')
        for identity, seed in zip(identities, seeds)]
    validated = [validate_prepared(job['prepared'], config) for job in jobs]
    x = torch.from_numpy(np.stack([item[0] for item in validated])).to(device)
    observed = torch.from_numpy(np.stack([item[2] for item in validated])).to(device)
    y = torch.from_numpy(np.stack([np.nan_to_num(item[1], nan=0.) for item in validated])).to(device)
    counts = observed.sum(1)
    cls = IndependentLSTM if model_name == 'lstm' else IndependentTCN
    model = cls(initialization_seeds, config.input_features, config.hidden_size).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                 foreach=False, fused=False)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.perf_counter()
    first_loss = None
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss_per_job = ((prediction-y).square()*observed).sum(1)/counts
        if first_loss is None:
            first_loss = loss_per_job.detach().cpu().numpy()
        # Sum, not mean over jobs: changing batch size cannot rescale a model's gradient.
        loss_per_job.sum().backward()
        _per_job_clip(parameters, config.gradient_clip)
        optimizer.step()
    with torch.no_grad():
        prediction = model(x)
        final_loss = (((prediction-y).square()*observed).sum(1)/counts).cpu().numpy()
        output = prediction.cpu().numpy()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    if not np.isfinite(output).all() or not np.isfinite(final_loss).all():
        raise FloatingPointError('Neural training diverged; do not replace with a hidden successful fallback.')
    return [{'prediction_all_std': output[i], 'training_summary': {
        'model': model_name, 'seed': seeds[i], 'job_id': identities[i],
        'initialization_seed': initialization_seeds[i],
        'config': asdict(config), 'n_observed': int(counts[i].item()),
        'initial_visible_mse_standardized': float(first_loss[i]),
        'final_visible_mse_standardized': float(final_loss[i]),
        'independent_local_parameters': True, 'cross_station_pretraining': False,
        'batch_jobs': len(jobs), 'batch_train_seconds': elapsed,
        'parameters_per_job': sum(parameter.numel() for parameter in parameters)//len(jobs),
        'device': str(device)}} for i in range(len(jobs))]


def fit_prepared_batch(prepared_list, model_name, seed, config=None, device='cuda'):
    return fit_many_jobs([{'prepared': p, 'seed': seed, 'job_id': p['case_id']} for p in prepared_list],
                         model_name, config, device)
