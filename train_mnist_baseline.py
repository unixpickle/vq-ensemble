from vq_ensemble.models import MNISTModel

import torch
import torch.nn.functional as F
import torch.optim as optim

from train_mnist import DEVICE, load_batches, create_datasets

EVAL_INTERVAL = 200
REDUCE_LR_STEPS = 500


def main():
    train_loader, test_loader = create_datasets()
    train_batches = load_batches(train_loader)
    test_batches = load_batches(test_loader)

    model = MNISTModel()
    model.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    step = 0
    while True:
        train_loss = model_loss(model, train_batches)
        test_loss = model_loss(model, test_batches)
        print('step: %d: train=%f test=%f' %
              (step, train_loss.item(), test_loss.item()))

        model.zero_grad()
        train_loss.backward()
        opt.step()

        if step == REDUCE_LR_STEPS:
            opt = optim.Adam(model.parameters(), lr=1e-4)

        step += 1
        if not step % EVAL_INTERVAL:
            evaluate(model, train_loader, 'train')
            evaluate(model, test_loader, 'test')


def model_loss(model, batches):
    ins, outs = next(batches)
    x = model(ins)
    return F.nll_loss(F.log_softmax(x, dim=-1), outs)


def evaluate(model, loader, dataset):
    num_correct = 0
    num_total = 0
    for input_batch, output_batch in loader:
        input_batch = input_batch.to(DEVICE)
        output_batch = output_batch.to(DEVICE)
        with torch.no_grad():
            model_outs = model(input_batch)
        classes = torch.argmax(model_outs, dim=-1)
        num_correct += torch.sum(classes == output_batch).item()
        num_total += input_batch.shape[0]
    print('Evaluation accuracy (%s): %.2f%%' % (dataset, 100 * num_correct / num_total))


if __name__ == '__main__':
    main()
