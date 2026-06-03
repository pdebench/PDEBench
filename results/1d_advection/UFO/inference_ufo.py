import torch
import numpy as np
import h5py

class UFO(torch.nn.Module):
    """Unitary Fourier Operator: FFT → phase multiplication → IFFT."""
    def __init__(self, num_points=1024):
        super().__init__()
        self.num_modes = num_points // 2 + 1
        self.phases = torch.nn.Parameter(torch.zeros(self.num_modes))

    def forward(self, x):
        X = x.shape[-1]
        x_hat = torch.fft.rfft(x, dim=-1)
        W = torch.exp(1j * self.phases.to(x.device))
        return torch.fft.irfft(x_hat * W, dim=-1, n=X)

def generate_predictions(initial_condition, weights_path, num_steps=201):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UFO(num_points=1024).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    x = torch.tensor(initial_condition[:, None, :], dtype=torch.float32).to(device)
    predictions = np.zeros((initial_condition.shape[0], num_steps, initial_condition.shape[1]))
    predictions[:, 0, :] = initial_condition
    with torch.no_grad():
        for t in range(1, num_steps):
            x = model(x)
            predictions[:, t, :] = x.squeeze(1).cpu().numpy()
    return predictions

if __name__ == '__main__':
    # Example usage (replace with actual initial condition)
    # init = load_your_initial_condition()
    # pred = generate_predictions(init, 'ufo_phases.pt')
    # save as HDF5
    pass