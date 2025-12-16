def split_by_label(X, y):
    X_min = {x for (x, label) in zip(X, y) if label == 1}
    X_maj = {x for (x, label) in zip(X, y) if label == 0}
    return X_min, X_maj


class Generator:
    def forward(self, noise):
        # map noise to a synthetic minority sample
        return x_fake


class DualCritic:
    def dom_head(self, x):
        # score for real-vs-synthetic minority (Wasserstein-style)
        return score

    def cls_head(self, x):
        # logit for minority-vs-majority classification
        return logit


def train_badc(X, y):
    X_min, X_maj = split_by_label(X, y)

    G = Generator()
    D = DualCritic()

    for step in range(NUM_TRAINING_STEPS):
        # ---- Critic update ----
        x_real = sample(X_min)
        z = sample_noise()
        x_fake = G.forward(z)

        # Wasserstein adversarial loss + gradient penalty (conceptually)
        L_wgan = critic_wasserstein_loss(D.dom_head, x_real, x_fake)

        # Supervised classification loss on real data (both classes)
        x_cls, y_cls = sample_labeled(X, y)
        L_cls = critic_classification_loss(D.cls_head, x_cls, y_cls)

        L_D = combine_losses(L_wgan, L_cls)
        update_parameters(D, L_D)

        # ---- Generator update ----
        z = sample_noise()
        x_fake = G.forward(z)

        # Fool domain head + enforce minority label under classification head
        L_G = generator_loss(D.dom_head, D.cls_head, x_fake)
        update_parameters(G, L_G)

    return G


def rebalance_source_with_badc(X, y):
    X_min, X_maj = split_by_label(X, y)
    G = train_badc(X, y)

    X_syn = []
    while len(X_min) + len(X_syn) < len(X_maj):
        z = sample_noise()
        x_new = G.forward(z)
        X_syn.append(x_new)

    X_bal = concatenate(X_maj, X_min, X_syn)
    y_bal = concatenate_labels(len(X_maj), len(X_min), len(X_syn))
    return X_bal, y_bal
