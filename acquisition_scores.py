import itertools
import math
import torch
import numpy as np

# Credit to Sebastian Farquhar
# @ https://github.com/SebFar/statistical_bias_in_active_learning/
def entropy_score(model, 
                  data_loader, # must not shuffle
                  n_samples, 
                  head=0):
    model.eval()
    scores = []
    with torch.no_grad():
        for data, _ in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            # Shape of samples is [variational_samples, batch_size, n_classes]
            samples = torch.stack([model.one_sample_forward(data, head) for _ in range(n_samples)])
            # mean_samples is log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            # Shape is [batch_size, n_classes]
            mean_samples = torch.logsumexp(samples, dim=0) - math.log(n_samples)
            # score is \Sum_{h \in C} 1/K \Sum_{i} p(y=h|X)q(w_i) log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            score = -torch.sum(mean_samples.exp() * mean_samples, dim=1)
            scores.append(score)
        scores = torch.cat(scores)
    return scores


def mutual_information(
    model,
    data_loader, # must not shuffle
    n_samples,
    head=0,
):
    # For consistent dropout this is critical to get the right behaviour
    model.eval()
    
    scores_N = []
    with torch.no_grad():
        for data, _ in data_loader:
            if torch.cuda.is_available():
                data = data.cuda()

            samples_V_N_K = torch.stack([model.one_sample_forward(data, head) for _ in range(n_samples)])
            average_entropy_N = -torch.sum(
                samples_V_N_K.exp() * samples_V_N_K, dim=2
            ).mean(0)

            mean_samples_N_K = torch.logsumexp(samples_V_N_K, dim=0) - math.log(
                n_samples
            )
            entropy_average_N = -torch.sum(
                mean_samples_N_K.exp() * mean_samples_N_K, dim=1
            )

            score_N = entropy_average_N - average_entropy_N
            scores_N.append(score_N)

    scores_N = torch.cat(scores_N)

    # Occassionally a noisy prediction on a very small MI might lead to negative score
    # We knock these up to 0
    scores_N[scores_N < 0] = 0.
    return scores_N.cpu()


def sample_proportionally(probability_masses, num_to_acquire):
    num_to_acquire = num_to_acquire
    assert len(probability_masses) >= num_to_acquire
    sampled_idxs = torch.multinomial(probability_masses, num_to_acquire)
    return sampled_idxs, probability_masses[sampled_idxs]


def sample_softmax(probability_masses, num_to_acquire, temperature):
    num_to_acquire = num_to_acquire
    assert len(probability_masses) >= num_to_acquire
    probability_masses = torch.nn.functional.softmax(
        temperature * probability_masses, dim=0
    )
    sampled_idxs = torch.multinomial(probability_masses, num_to_acquire)
    return sampled_idxs, probability_masses[sampled_idxs]