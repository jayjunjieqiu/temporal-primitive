from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from cuml.manifold import TSNE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--perplexity", type=float, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--init", choices=["random", "pca"], default="random")
    args = parser.parse_args()

    x_pre = np.load(args.input).astype(np.float32, copy=False)
    effective_perplexity = float(min(args.perplexity, max(5, (len(x_pre) - 1) // 3)))

    t0 = time.perf_counter()
    n_neighbors = max(90, int(np.ceil(3 * effective_perplexity)) + 1)
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        n_neighbors=n_neighbors,
        max_iter=int(args.max_iter),
        init=args.init,
        random_state=int(args.seed),
        metric="euclidean",
        method="fft",
        learning_rate_method="adaptive",
        output_type="numpy",
    )
    x_tsne = reducer.fit_transform(x_pre).astype(np.float32, copy=False)
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x_tsne)
    Path(args.summary).write_text(
        json.dumps(
            {
                "backend": "cuml_tsne",
                "input_shape": list(x_pre.shape),
                "output_shape": list(x_tsne.shape),
                "perplexity": effective_perplexity,
                "n_neighbors": n_neighbors,
                "max_iter": int(args.max_iter),
                "init": args.init,
                "fit_sec": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
