import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch

from model import FNO1d


def generate_wave(
        resolution: int,
        wave: str = 'parabola'
    ) -> torch.Tensor:

    x = torch.arange(0, resolution) / resolution
    if wave == 'parabola':
        w = x * (1 - x)
    elif wave == 'square':
        w = torch.zeros(resolution)
        w[:resolution // 2] = 1.
    else:
        raise Exception()
    
    grid: torch.Tensor = torch.linspace(0.0, 1.0, resolution)
    return torch.stack([w, grid], dim = -1).unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=str)
    parser.add_argument('-p', type=int, default=16)
    parser.add_argument('-T', type=int, default=4)

    args = parser.parse_args()
    wave = args.wave
    p = args.p
    T = args.T

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    np.random.seed(0)

    w: torch.Tensor = generate_wave(2 ** p, wave=wave)

    histories: list[list[list[float]]] = []
    for t in range(T):
        models = [
            FNO1d(
                modes = 16,
                width = 64,
                n_layers = t + 1
            ) for _ in range(100)
        ]
        history: list[list[float]] = []
        for n in range(7, p - 1):
            row: list[float] = []
            for model in models:
                model.eval()
                input_1 = w[:, ::(2 ** (p - 6)),     :]
                input_2 = w[:, ::(2 ** (p - n)), :]
                sub_step = 2 ** (n - 6)
                y = torch.sum(torch.abs(model(input_1) - model(input_2)[:, ::sub_step, :])) / 2 ** 6
                y = y.item()
                row.append(y)
            history.append(row)
        histories.append(history)

        x = np.arange(1, p - 7)
        history = np.array(histories[t])
        plt.plot(
            x,
            history.max(axis = 1),
            linestyle = 'dotted',
            color = f'C{t}'
        )
        plt.plot(
            x,
            history.min(axis = 1),
            linestyle = 'dotted',
            color = f'C{t}'
        )
        plt.fill_between(
            x,
            history.min(axis = 1),
            history.max(axis = 1),
            color = f'C{t}',
            alpha = 0.1
        )
        plt.plot(
            x,
            history.mean(axis = 1),
            color = f'C{t}',
            label = rf'T={t + 1}',
            linewidth = 1,
            marker = 'o'
        )

    plt.yscale('log')
    plt.grid(True, which = 'both')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$|\Psi^M_\theta(a)(x)-\Psi^N_\theta(a)(x)|$')
    plt.legend()
    plt.tight_layout()
    plt.margins(x = 0, y = 0)

    plt.show()
