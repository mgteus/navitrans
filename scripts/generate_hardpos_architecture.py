"""Render the trained HardPos architecture with VisualTorch.

Run from the repository root with .nav/bin/python scripts/generate_hardpos_architecture.py.
"""
from __future__ import annotations
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import torch
from torch import nn
import visualtorch

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "navitran_vizz_reboot_gpt_squared_25_span.ipynb"
CHECKPOINT = ROOT / "legacy/checkpoints/HardPos_/gpt_model_HardPos_.pt"
OUTPUT = ROOT / "outputs" / "visualtorch"

def load_model() -> nn.Module:
    """Use the class definition and parameters of the dissertation checkpoint."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"torch": torch, "nn": nn, "math": math}
    for index in (21, 23):
        exec("".join(notebook["cells"][index]["source"]), namespace)
    model = namespace["TrajectoryBERT_HardPos"](
        vocab_size=948, embed_size=16, num_layers=1, num_heads=1,
        hidden_dim=32, dropout=0.2, pad_token_id=947, max_len=100, causal=True,
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    return model.cpu().eval()

def main() -> None:
    model = load_model()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_shape = (1, 10)
    common = dict(input_dtype=torch.long, palette="okabe_ito", background_fill="white")
    graph = dict(style="graph", show_neurons=False, show_arrows=True,
                 show_dimension=True, layer_spacing=150, legend=True)
    visualtorch.render(model, input_shape, to_file=str(OUTPUT / "navitrans_graph.png"),
                       **graph, **common)
    visualtorch.render(model, input_shape, style="flow", show_dimension=True,
                       spacing=65, min_xy=80, max_xy=220, scale_xy=0.5, legend=True, to_file=str(OUTPUT / "navitrans_flow.png"), **common)
    visualtorch.animate(model, input_shape, to_file=str(OUTPUT / "navitrans_architecture.gif"),
                        frame_duration=450, final_hold_duration=1500, loop=True,
                        **graph, **common)
    metadata = {
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "model_source": NOTEBOOK.name,
        "input_shape": input_shape,
        "input_dtype": "torch.long",
        "visualtorch_version": version("visualtorch"),
        "torch_version": torch.__version__,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
