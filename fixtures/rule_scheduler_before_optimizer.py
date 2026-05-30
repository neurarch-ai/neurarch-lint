"""Fixture: lr scheduler stepped before the optimizer (scheduler-step-before-optimizer)."""
import torch


def train(model, optimizer, scheduler, loader, loss_fn):
    for batch, target in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, target)
        loss.backward()
        # WRONG ORDER: the scheduler steps before the optimizer, which skips
        # the first learning-rate value.
        scheduler.step()
        optimizer.step()
