import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.custom_load_data import create_data_loader
from utils.model.mymodel import Unet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for input_data, target, fnames, slices in data_loader:
            input_data = input_data.cuda(non_blocking=True)
            output = model(input_data)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = Unet()
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location=device)
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loader(data_path=args.data_path, batch_size=args.batch_size, shuffle=False, transform=None)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
