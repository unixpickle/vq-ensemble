from vq_ensemble.meta_models import Encoder, Refiner
from vq_ensemble.models import MNISTModel

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

INNER_BATCH_SIZE = 128
META_BATCH_SIZE = 4
NUM_STAGES = 20
INNER_LR = 1e-3
META_LR = 1e-4
EVAL_INTERVAL = 100


def main():
    train_loader, test_loader = create_datasets()
    train_batches = load_batches(train_loader)
    test_batches = load_batches(test_loader)

    model = MNISTModel()
    meta_model = Encoder(Refiner(model.param_size(), NUM_STAGES))
    model.to(DEVICE)
    meta_model.to(DEVICE)

    opt = optim.Adam(meta_model.parameters(), lr=META_LR)
    step = 0
    while True:
        train_loss, train_inner_loss = create_meta_batch(meta_model, model, train_batches)
        test_loss, test_inner_loss = create_meta_batch(meta_model, model, test_batches)
        print('step: %d: train=%f test=%f train_inner=%f test_inner=%f' %
              (step, train_loss.item(), test_loss.item(), train_inner_loss, test_inner_loss))

        meta_model.zero_grad()
        train_loss.backward()
        opt.step()

        step += 1
        if not step % EVAL_INTERVAL:
            evaluate(meta_model, model, train_loader, 'train')
            evaluate(meta_model, model, test_loader, 'test')


def evaluate(meta_model, model, loader, dataset):
    weights = meta_model.decode(meta_model.random_latents(1))[0, -1]
    model.set_parameters(weights)
    num_correct = 0
    num_total = 0
    for input_batch, output_batch in loader:
        input_batch = input_batch.to(DEVICE)
        output_batch = output_batch.to(DEVICE)
        outputs = model(input_batch)
        classes = torch.argmax(outputs, dim=-1)
        num_correct += torch.sum(classes == output_batch).item()
        num_total += input_batch.shape[0]
    print('Evaluation accuracy (%s): %.2f%%' % (dataset, 100 * num_correct / num_total))


def create_meta_batch(meta_model, model, inner_batches):
    latents = meta_model.random_latents(META_BATCH_SIZE)
    sample_seqs = meta_model.decode(latents)
    parameters = sample_seqs[:, -1]
    init_losses = []
    all_targets = []
    for i in range(META_BATCH_SIZE):
        inner_params = parameters[i]
        model.set_parameters(inner_params)
        inner_batch, inner_targets = next(inner_batches)
        output = F.log_softmax(model(inner_batch), dim=-1)
        loss = F.nll_loss(output, inner_targets)
        init_losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.detach().add_(-p.grad * INNER_LR)
        all_targets.append(model.get_parameters())
    targets = torch.stack(all_targets, dim=0)
    meta_loss = torch.mean(torch.pow(targets[:, None] - parameters, 2))
    meta_loss *= model.param_size()
    return meta_loss, np.mean(init_losses)


def load_batches(dataset):
    while True:
        for ins, outs in dataset:
            yield ins.to(DEVICE), outs.to(DEVICE)


def create_datasets():
    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True, transform=transform),
        batch_size=INNER_BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transform),
        batch_size=INNER_BATCH_SIZE, shuffle=True, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    main()
