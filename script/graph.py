"""
  Filename      [ graph.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Visualize model graph with tensorboardX ]
"""

import torch
from model.At_model import Dense
from tensorboardX import SummaryWriter

def main():
    model = Dense()
    dummy_input = torch.rand(1, 3, 512, 512)

    with SummaryWriter(comment='Dense') as writer:
        writer.add_graph(model, dummy_input)
    
    return

if __name__ == "__main__":
    main()