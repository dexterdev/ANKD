"""Saving and loading checkpoints.

Everything is written as a container -- model_state plus metadata -- and `load`
tolerates the shapes that turn up in practice: a bare state_dict, a different
container key, a 'module.' prefix from DataParallel, and the 'backbone.' prefix
that the normalization wrapper adds but a bare-backbone checkpoint lacks.
"""

from __future__ import annotations

import os

import torch

_CONTAINER_KEYS = ("model_state", "state_dict", "model", "net", "weights", "teacher")
_META_KEYS = ("val_acc", "train_acc", "acc", "best_acc", "epoch", "epochs",
              "temperature", "arch")


def _strip(prefix):
    return lambda k: k[len(prefix):] if k.startswith(prefix) else k


_KEY_FIXES = [
    ("as-is", lambda k: k),
    ("add 'backbone.'", lambda k: f"backbone.{k}"),
    ("strip 'backbone.'", _strip("backbone.")),
]


def save(model, path: str, **meta) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)


def _candidates(obj):
    if not isinstance(obj, dict):
        return
    if any(torch.is_tensor(v) for v in obj.values()):
        yield obj
    for key in _CONTAINER_KEYS:
        value = obj.get(key)
        if isinstance(value, dict) and any(torch.is_tensor(t) for t in value.values()):
            yield value


def load(model, path: str, device) -> dict:
    """Load `path` into `model`; returns whatever metadata the file carried.

    Raises with the keys it actually saw when nothing matches, rather than a wall
    of missing-key output.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                                   # torch too old for weights_only
        obj = torch.load(path, map_location=device)

    meta = {k: obj[k] for k in _META_KEYS if isinstance(obj, dict) and k in obj}
    want = model.state_dict()
    best = (-1, None, None)

    for state in _candidates(obj):
        state = {_strip("module.")(k): v for k, v in state.items()}
        for name, fix in _KEY_FIXES:
            cand = {fix(k): v for k, v in state.items()}
            overlap = len(set(want) & set(cand))
            if overlap > best[0]:
                best = (overlap, name, cand)
            if set(cand) == set(want):
                bad = {k: (tuple(v.shape), tuple(want[k].shape))
                       for k, v in cand.items() if v.shape != want[k].shape}
                if bad:
                    k, (got, exp) = next(iter(bad.items()))
                    raise RuntimeError(
                        f"{path} matches the architecture but not its shapes "
                        f"({len(bad)} tensor(s) differ, e.g. {k}: checkpoint {got} vs "
                        f"model {exp}). A classifier mismatch means the checkpoint was "
                        f"trained on a different dataset.")
                model.load_state_dict(cand)
                meta["_matched"] = name
                return meta

    overlap, name, cand = best
    raise RuntimeError(
        f"Could not match {path} to this model.\n"
        f"  best attempt ({name}) matched {overlap}/{len(want)} keys\n"
        f"  checkpoint keys (first 5): {sorted(cand)[:5] if cand else 'none found'}\n"
        f"  model keys (first 5):      {sorted(want)[:5]}\n"
        f"  Unrelated names mean the checkpoint is a different architecture.")
