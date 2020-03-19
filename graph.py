"""
  Filename      [ graph.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Visualize model graph with tensorboardX ]
"""

import torch
from model.At_model import Dense
from model.AtJ_At import AtJ
from tensorboardX import SummaryWriter

def main():
    dense = Dense()
    # atj = AtJ()

    dummy_input = torch.rand(1, 3, 512, 512)

    with SummaryWriter(comment='ModelGraph') as writer:
        writer.add_graph(dense, dummy_input, verbose=True, omit_useless_nodes=False)
        # writer.add_graph(atj, dummy_input, verbose=True, omit_useless_nodes=False)

    return

if __name__ == "__main__":
    main()
