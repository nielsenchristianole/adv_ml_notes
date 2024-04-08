# Programming excercise: Shallow embedding
# import numpy as np
# a = np.array([
# [0, 0, 1, 1, 0, 1, 0,],
# [0, 0, 0, 0, 1, 1, 1,],
# [1, 0, 0, 1, 0, 1, 0,],
# [1, 0, 1, 0, 0, 1, 0,],
# [0, 1, 0, 0, 0, 1, 1,],
# [1, 1, 1, 1, 1, 0, 1,],
# [0, 1, 0, 0, 1, 1, 0,]])
# D = a.sum(1)
# 1 / (D * (D - 1)) * np.diag(np.linalg.matrix_power(a, 3))

# # b1
# P = a / a.sum(1)[:, None]
# pi = np.array([1, 0, 0, 0, 0, 0, 0])
# n_step = 1
# pi_final = pi @ np.linalg.matrix_power(P, n_step)


# # b2
# t: int
# np.linalg.matrix_power(a, t)




# Import libraries
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch


# Shallow node embedding
class Shallow(torch.nn.Module):
    '''Shallow node embedding

    Args: 
        n_nodes (int): Number of nodes in the graph
        embedding_dim (int): Dimension of the embedding
    '''
    def __init__(self, n_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, rx, tx):
        '''Returns the probability of a links between nodes in lists rx and tx'''
        return torch.sigmoid((self.embedding.weight[rx]*self.embedding.weight[tx]).sum(1) + self.bias)


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, max_step: int, idx_pairs: torch.Tensor, target: torch.Tensor, criterion: torch.nn.Module):

    for idx in (progress_bar := tqdm(range(max_step), leave=False)):

        pred = model(idx_pairs[0], idx_pairs[1])
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 300 == 0:
            grad_length = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]))
            if grad_length < 1e-4: # early stopping
                break
            progress_bar.set_description(f'Loss = {loss.item():.5f}, Grad = {grad_length:.5f}')
    progress_bar.close()


def test(model: torch.nn.Module, idx_pairs: torch.Tensor, target: torch.Tensor) -> float:
    '''Returns the accuracy of the model'''
    pred = (model(idx_pairs[0], idx_pairs[1]) > 0.5).float()
    return (pred == target).float().mean().item()


def main(device: str, lr: float, max_step: int, n_splits: int, embedding_dim_space: list[int]):
    A = torch.load('data.pt', map_location=device)
    n_nodes = A.shape[0]
    idx_all_pairs = torch.triu_indices(n_nodes, n_nodes, 1, device=device)
    target = A[idx_all_pairs[0], idx_all_pairs[1]]

    cross_entropy = torch.nn.BCELoss().to(device)

    acc = torch.empty((n_splits, len(embedding_dim_space)))
    for split_idx, (train_idx, test_idx) in tqdm(enumerate(KFold(n_splits=n_splits).split(idx_all_pairs[0])), total=n_splits, desc='Splits'):
        train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)

        train_idx_pairs = idx_all_pairs[:, train_idx]
        test_idx_pairs = idx_all_pairs[:, test_idx]

        train_target = target[train_idx]
        test_target = target[test_idx]

        for emb_idx, emb_dim in (p_bar := tqdm(enumerate(embedding_dim_space), total=len(embedding_dim_space), desc='Hyper params', leave=False)):
            p_bar.set_description(f'emb_dim {emb_dim}')

            model = Shallow(n_nodes, emb_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train(model, optimizer, max_step, train_idx_pairs, train_target, cross_entropy)
            accuracy = test(model, test_idx_pairs, test_target)

            acc[split_idx, emb_idx] = accuracy


    best_hyperparams_idx = acc.mean(0).argmax()
    best_hyperparams = embedding_dim_space[best_hyperparams_idx]

    print(f"Training best model, with embedding dimension: {best_hyperparams} and accuracy: {acc.mean(0).max():.5f}")

    model = Shallow(n_nodes, best_hyperparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, max_step, idx_all_pairs, target, cross_entropy)

    # save model statedict
    torch.save(model.state_dict(), 'model.pt')
    torch.save(acc.detach().cpu(), 'acc.pt')

    link_probability = model(idx_all_pairs[0], idx_all_pairs[1])
    torch.save(link_probability, 'link_probability.pt')


    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Shallow embedding")
    parser.add_argument('--device', type=str, help='cpu or cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--max_step', default=100000, type=int, help='maximum number of steps')
    parser.add_argument('--n_splits', default=10, type=int, help='number of splits for cross validation')
    parser.add_argument('--min_embedding_dim', default=3, type=int, help='minimum embedding dimension')
    parser.add_argument('--max_embedding_dim', default=7, type=int, help='maximum embedding dimension')
    parser.add_argument('--embedding_dim_space', nargs='+', type=int, help='list of embedding dimensions will override min_embedding_dim and max_embedding_dim')
    parser.add_argument('--verbose', action='store_true', help='print verbose')
    args = parser.parse_args()
    

    device = args.device or 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = args.lr
    max_step = args.max_step
    n_splits = args.n_splits
    embedding_dim_space = args.embedding_dim_space or list(range(args.min_embedding_dim, args.max_embedding_dim + 1))


    main(device, lr, max_step, n_splits, embedding_dim_space)