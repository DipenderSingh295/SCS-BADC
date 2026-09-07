import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        in_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class DualHeadCritic(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.domain_head = nn.Linear(in_dim, 1)
        self.class_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        domain_score = self.domain_head(features).squeeze(1)
        class_logit = self.class_head(features).squeeze(1)

        return domain_score, class_logit


class BADC:
    def __init__(
        self,
        input_dim,
        latent_dim,
        generator_hidden,
        critic_hidden,
        lr=1e-4,
        lambda_gp=10.0,
        alpha=1.0,
        beta=1.0,
        critic_updates=5,
        adam_betas=(0.9, 0.999),
        device=None,
    ):
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.lambda_gp = float(lambda_gp)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.critic_updates = int(critic_updates)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dims=generator_hidden,
            output_dim=self.input_dim,
        ).to(self.device)

        self.critic = DualHeadCritic(
            input_dim=self.input_dim,
            hidden_dims=critic_hidden,
        ).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=adam_betas,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
            betas=adam_betas,
        )

    def _to_tensor(self, X):
        return torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

    def _gradient_penalty(self, real, fake):
        batch_size = real.shape[0]

        interpolation_weight = torch.rand(
            batch_size,
            1,
            device=self.device,
        )

        interpolated = (
            interpolation_weight * real
            + (1.0 - interpolation_weight) * fake
        )

        interpolated.requires_grad_(True)

        domain_score, _ = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=domain_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(domain_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)

        return (
            gradients.norm(2, dim=1) - 1.0
        ).pow(2).mean()

    def fit(
        self,
        X,
        y,
        iterations,
        batch_size,
        seed=0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        X = self._to_tensor(X)
        y = self._to_tensor(y).view(-1)

        defective = X[y == 1]

        if defective.shape[0] == 0:
            raise ValueError(
                "At least one defective-class instance is required."
            )

        for _ in range(int(iterations)):
            for _ in range(self.critic_updates):
                defective_index = torch.randint(
                    0,
                    defective.shape[0],
                    (batch_size,),
                    device=self.device,
                )

                source_index = torch.randint(
                    0,
                    X.shape[0],
                    (batch_size,),
                    device=self.device,
                )

                real_defective = defective[defective_index]
                source_batch = X[source_index]
                source_labels = y[source_index]

                z = torch.randn(
                    batch_size,
                    self.latent_dim,
                    device=self.device,
                )

                synthetic = self.generator(z).detach()

                real_score, _ = self.critic(real_defective)
                synthetic_score, _ = self.critic(synthetic)
                _, class_logits = self.critic(source_batch)

                gradient_penalty = self._gradient_penalty(
                    real_defective,
                    synthetic,
                )

                wasserstein_loss = (
                    synthetic_score.mean()
                    - real_score.mean()
                    + self.lambda_gp * gradient_penalty
                )

                classification_loss = (
                    F.binary_cross_entropy_with_logits(
                        class_logits,
                        source_labels,
                    )
                )

                critic_loss = (
                    wasserstein_loss
                    + self.alpha * classification_loss
                )

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()

            z = torch.randn(
                batch_size,
                self.latent_dim,
                device=self.device,
            )

            synthetic = self.generator(z)

            synthetic_score, synthetic_class_logits = (
                self.critic(synthetic)
            )

            distribution_loss = -synthetic_score.mean()

            classification_guidance = (
                F.binary_cross_entropy_with_logits(
                    synthetic_class_logits,
                    torch.ones_like(synthetic_class_logits),
                )
            )

            generator_loss = (
                distribution_loss
                + self.beta * classification_guidance
            )

            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            self.generator_optimizer.step()

        return self

    def generate(self, n_samples, batch_size=1024):
        n_samples = int(n_samples)

        if n_samples <= 0:
            return np.empty(
                (0, self.input_dim),
                dtype=np.float32,
            )

        self.generator.eval()

        generated = []

        with torch.no_grad():
            remaining = n_samples

            while remaining > 0:
                current_batch = min(
                    batch_size,
                    remaining,
                )

                z = torch.randn(
                    current_batch,
                    self.latent_dim,
                    device=self.device,
                )

                generated.append(
                    self.generator(z).cpu().numpy()
                )

                remaining -= current_batch

        self.generator.train()

        return np.vstack(generated)

    def balance(self, X, y, generation_batch_size=1024):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)

        defective_count = int(np.sum(y == 1))
        nondefective_count = int(np.sum(y == 0))

        required = max(
            0,
            nondefective_count - defective_count,
        )

        if required == 0:
            return X.copy(), y.copy()

        synthetic = self.generate(
            required,
            batch_size=generation_batch_size,
        )

        synthetic_labels = np.ones(
            required,
            dtype=y.dtype,
        )

        X_balanced = np.vstack(
            [X, synthetic]
        )

        y_balanced = np.concatenate(
            [y, synthetic_labels]
        )

        return X_balanced, y_balanced

    def fit_balance(
        self,
        X,
        y,
        iterations,
        batch_size,
        seed=0,
        generation_batch_size=1024,
    ):
        self.fit(
            X=X,
            y=y,
            iterations=iterations,
            batch_size=batch_size,
            seed=seed,
        )

        return self.balance(
            X=X,
            y=y,
            generation_batch_size=generation_batch_size,
        )
