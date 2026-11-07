from neuralop.models import FNO

import os
import math
import argparse
import pandas as pd
import torch

from tqdm import tqdm
from matplotlib import pyplot as plt


def generate_wave(resolution: int) -> torch.Tensor:
    x = torch.arange(0, resolution) / resolution
    pi = math.pi

    w = (
        0.25 * torch.sin(20 * pi * x)
        + ((1 + torch.cos(2 * pi * x)) / 2) ** 20
    )

    return w.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_n', type=int, default=5)
    parser.add_argument('--max_n', type=int, default=15)
    parser.add_argument('-m', '--n_modes', type=int, default=16)
    parser.add_argument('-d', '--results_dir', type=str, default='results')
    parser.add_argument('-N', '--num_models', type=int, default=100)
    args = parser.parse_args()

    min_n = args.min_n
    max_n = args.max_n
    n_modes = args.n_modes
    results_dir = args.results_dir
    num_models = args.num_models

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using {device} ...')

    wave = generate_wave(2 ** max_n).to(device)

    for t in range(4):
        df = pd.DataFrame(
            index=range(num_models),
            columns=range(min_n, max_n),
            dtype=float
        )

        for i in tqdm(range(num_models)):
            model = FNO(
                n_modes=(n_modes,),
                hidden_channels=16,
                in_channels=1,
                out_channels=1,
                n_layers=t + 1
            ).to(device)

            model.eval()

            out_dict = {}

            with torch.no_grad():
                for n in range(min_n, max_n + 1):
                    s = 2 ** (max_n - n)
                    wave_n = wave[:, ::s].unsqueeze(1)

                    out = model(wave_n)
                    out_dict[n] = out[0, 0].cpu()

            for n in range(min_n, max_n):

                x = out_dict[n]
                y = out_dict[n + 1]

                average = torch.sum(
                    torch.abs(x - y[::2])
                ) / (2 ** n)

                df.loc[i, n] = average.item()

        os.makedirs(results_dir, exist_ok=True)

        df.to_csv(
            os.path.join(results_dir, f'results_{t+1}.csv'),
            index=False
        )

        print(df)

        plt.plot(
            range(min_n, max_n),
            df.mean(axis=0).values,
            color=f'C{t}',
            marker='o',
            label=f'T = {t+1}'
        )
        plt.plot(
            range(min_n, max_n),
            df.max(axis=0).values,
            color=f'C{t}',
            linestyle='dotted',
        )
        plt.plot(
            range(min_n, max_n),
            df.min(axis=0).values,
            color=f'C{t}',
            linestyle='dotted',
        )
        plt.fill_between(
            range(min_n, max_n),
            df.min(axis=0).values,
            df.max(axis=0).values,
            color=f'C{t}',
            alpha=0.1,
        )

    plt.plot(
        range(min_n, max_n),
        [1 / (2 ** n) for n in range(min_n, max_n)],
        color='black',
        label = r'$2^{-n}$',
        marker = 'x'
    )

    plt.grid()
    plt.yscale('log')
    plt.ylim(1e-8, 1e-2)
    plt.grid(True, which = 'both')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$|\Psi^M_\theta(a)(x)-\Psi^N_\theta(a)(x)|$')
    plt.legend()
    plt.tight_layout()
    plt.margins(x = 0, y = 0.25)
    plt.show()
