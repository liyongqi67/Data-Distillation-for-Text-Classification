import logging
import math
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train_steps_inplace(state, models, steps, params=None, callback=None):
    if isinstance(models, torch.nn.Module):
        models = [models]
    if params is None:
        params = [m.get_param() for m in models]

    for i, (data, label, lr) in enumerate(steps):
        if callback is not None:
            callback(i, params)

        data = data.detach()
        label = label.detach()
        lr = lr.detach()

        for model, w in zip(models, params):
            model.train()  # callback may change model.training so we set here
            output = model.forward_with_param(data, w)
            loss = nn.NLLLoss()(output, label)
            loss.backward(lr.squeeze())
            with torch.no_grad():
                w.sub_(w.grad)
                w.grad = None

    if callback is not None:
        callback(len(steps), params)

    return params

# NOTE [ Evaluation Result Format ]
#
# Result is always a 3-tuple, containing (test_step_indices, accuracies, losses):
#
# - `test_step_indices`: an int64 vector of shape [NUM_STEPS].
# - `accuracies`:
#   + for mode != 'distill_attack', a matrix of shape [NUM_STEPS, NUM_MODELS].
#   + for mode == 'distill_attack', a tensor of shape
#       [NUM_STEPS, NUM_MODELS x NUM_CLASSES + 3], where the last dimensions
#       contains
#         [overall acc, acc w.r.t. modified labels,
#          class 0 acc, class 1 acc, ...,
#          ratio of attack_class predicted as target_class]
# - `losses`: a matrix of shape [NUM_STEPS, NUM_MODELS]


# See NOTE [ Evaluation Result Format ] for output format
def evaluate_models(state, models, param_list=None, test_all=False, test_loader_iter=None):
    n_models = len(models)
    device = state.device
    num_classes = state.num_classes
    corrects = np.zeros(n_models, dtype=np.int64)
    losses = np.zeros(n_models)

    total = np.array(0, dtype=np.int64)

    if test_all or test_loader_iter is None:  # use raw full iter for test_all
        test_loader_iter = state.test_loader_iter
    for model in models:
        model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader_iter):


            data, target = data.to(device), target.to(device)


            for k, model in enumerate(models):
                if param_list is None or param_list[k] is None:
                    output = model(data)
                else:
                    output = model.forward_with_param(data, param_list[k])

                if num_classes == 2:
                    pred = (output > 0.5).to(target.dtype).view(-1)
                else:
                    pred = output.argmax(1)  # get the index of the max log-probability

                correct_list = pred == target
                losses[k] += nn.NLLLoss()(output, target).item()  # sum up batch loss

                corrects[k] += correct_list.sum().item()

            total += output.size(0)

        losses /= total

    accs = corrects / total

    return accs, losses


def fixed_width_fmt(num, width=4, align='>'):
    if math.isnan(num):
        return '{{:{}{}}}'.format(align, width).format(str(num))
    return '{{:{}0.{}f}}'.format(align, width).format(num)[:width]


def _desc_step(state, steps, i):
    if i == 0:
        return 'before steps'
    else:
        lr = steps[i - 1][-1]
        return 'step {:2d} (lr={})'.format(i, fixed_width_fmt(lr.sum().item(), 6))


# See NOTE [ Evaluation Result Format ] for output format
def format_stepwise_results(state, steps, info, res):
    accs = res[1] * 100
    losses = res[2]
    acc_mus = accs.mean(1)
    acc_stds = accs.std(1, unbiased=True)
    loss_mus = losses.mean(1)
    loss_stds = losses.std(1, unbiased=True)

    def format_into_line(*fields, align='>'):
        single_fmt = '{{:{}24}}'.format(align)
        return ' '.join(single_fmt.format(f) for f in fields)

    msgs = [format_into_line('STEP', 'ACCURACY', 'LOSS', align='^')]
    acc_fmt = '{{: >8.4f}} {}{{: >5.2f}}%'.format(u'±')
    loss_fmt = '{{: >8.4f}} {}{{: >5.2f}}'.format(u'±')
    tested_steps = set(res[0].tolist())
    for at_step, acc_mu, acc_std, loss_mu, loss_std in zip(res[0], acc_mus, acc_stds, loss_mus, loss_stds):


        desc = _desc_step(state, steps, at_step)
        loss_str = loss_fmt.format(loss_mu, loss_std)
        acc_mu = acc_mu.view(-1)  # into vector
        acc_std = acc_std.view(-1)  # into vector
        acc_str = acc_fmt.format(acc_mu[0], acc_std[0])
        msgs.append(format_into_line(desc, acc_str, loss_str))


    return '{} test results:\n{}'.format(info, '\n'.join(('\t' + m) for m in msgs))


def infinite_iterator(iterable):
    while True:
        yield from iter(iterable)


# See NOTE [ Evaluation Result Format ] for output format
def evaluate_steps(state, steps, prefix, details='', test_all=False, test_at_steps=None, log_results=True):
    models = state.test_models
    n_steps = len(steps)

    if test_at_steps is None:
        test_at_steps = [0, n_steps]
    else:
        test_at_steps = [(x if x >= 0 else n_steps + 1 + x) for x in test_at_steps]

    test_at_steps = set(test_at_steps)
    N = len(test_at_steps)

    # cache test dataloader iter
    if test_all:
        test_loader_iter = None
    else:
        test_loader_iter = state.test_loader_iter

    test_nets_desc = '{} {} nets'.format(len(models), "111")

    def _evaluate_steps(comment, reset):  # returns Tensor [STEP x MODEL]
        if len(comment) > 0:
            comment = '({})'.format(comment)
            pbar_desc = prefix + ' ' + comment
        else:
            pbar_desc = prefix

        if log_results:
            pbar = tqdm(total=N, desc=pbar_desc)

        at_steps = []
        accs = []      # STEP x MODEL (x CLASSES)
        totals = []    # STEP x MODEL (x CLASSES)
        losses = []    # STEP x MODEL

        if reset:
            params = [m.reset(state, inplace=False) for m in models]
        else:
            params = [m.get_param(clone=True) for m in models]

        def test_callback(at_step, params):
            if at_step not in test_at_steps:  # not test_all and
                return

            acc, loss = evaluate_models(state, models, params, test_all=test_all,
                                        test_loader_iter=test_loader_iter)

            at_steps.append(at_step)
            accs.append(acc)
            losses.append(loss)
            if log_results:
                pbar.update()

        params = train_steps_inplace(state, models, steps, params, callback=test_callback)
        if log_results:
            pbar.close()

        at_steps = torch.as_tensor(at_steps, device=state.device)  # STEP
        accs = torch.as_tensor(accs, device=state.device)          # STEP x MODEL (x CLASS)
        losses = torch.as_tensor(losses, device=state.device)      # STEP x MODEL
        return at_steps, accs, losses

    if log_results:
        logging.info('')
        logging.info('{} {}{}:'.format(prefix, details, ' (test ALL)' if test_all else ''))
    res = _evaluate_steps(test_nets_desc, reset=(state.test_nets_type == 'unknown_init'))



    if log_results:
        result_title = '{} {} ({})'.format(prefix, details, test_nets_desc)
        logging.info(format_stepwise_results(state, steps, result_title, res))
        logging.info('')
    return res



