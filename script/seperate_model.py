"""
  Filename      [ seperate_model.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Seperate the model parameter out from CKPT ]
"""
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    opt = parser.parse_args()

    model_params = torch.load(opt.model)["model"]
    torch.save(model_params, os.path.join(os.path.dirname(opt.model), "AtJ_DH_MODEL.pth"))

if __name__ == "__main__":
    main()
