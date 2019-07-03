# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#   
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action

from numpy import array, diag, exp, matmul, mod
from scipy.special import expit as sigmoid
from numpy.random.mtrand import RandomState

env_args = {
    'num_products': 100,
    'random_seed': 123,
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25
}

env_1_args = {
    **env_args,
    **{
        'K': 5,  # latent dimension size
        'sigma_omega_initial': 1,
        'sigma_omega': 0.1,
        'number_of_flips': 0,  # how different is organic from bandit
        'sigma_mu_organic': 3,
        'change_omega_for_bandits': False
    }
}

# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2


# Environment definition.
class RecoEnv():

    def __init__(self, config):
        self.first_step = True
        self.state = None
        self.empty_sessions = []

        self.config = config
        self.rng = RandomState(self.config.random_seed)
        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

    # Static function for squashing values between 0 and 1.
    def f(self, mat, offset=5):
        return sigmoid(mat - offset)

    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave.
        self.state_transition = array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Initialise Gamma for all products (Organic).
        self.Gamma = self.rng.normal(
            size=(self.config.num_products, self.config.K)
        )

        # Initialise mu_organic.
        self.mu_organic = self.rng.normal(
            0, self.config.sigma_mu_organic,
            size=(self.config.num_products)
        )

        # Initialise beta, mu_bandit for all products (Bandit).
        self.generate_beta(self.config.number_of_flips)

    def reset_random_seed(self):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed)

    # Create a new user.
    def reset(self):
        # Current state.
        self.first_step = True
        self.state = organic  # Manually set first state as Organic.

        self.omega = self.rng.normal(
            0, self.config.sigma_omega_initial, size=(self.config.K, 1)
        )

    def generate_organic_sessions(self):
        # Initialize session.
        session = []
        while self.state == organic:
            # Add next product view.
            session.append(self.sample_product_view())
            # Update markov state.
            self.update_state()
        return session

    def step(self, action_id):
        # No information to return.
        info = {}

        if self.first_step:
            assert (action_id is None)
            self.first_step = False
            sessions = self.generate_organic_sessions()
            return sessions, None, None, info

        assert (action_id is not None)
        # Calculate reward from action.
        reward = self.draw_click(action_id)

        self.update_state()

        if reward == 1:
            self.state = organic  # Clicks are followed by Organic.

        # Markov state dependent logic.
        if self.state == organic:
            sessions = self.generate_organic_sessions()
        else:
            sessions = self.empty_sessions

        # Update done flag.
        done = True if self.state == stop else False

        return sessions, reward, done, info

    # Update user state to one of (organic, bandit, leave) and their omega (latent factor).
    def update_state(self):
        self.state = self.rng.choice(3, p=self.state_transition[self.state, :])

        # And update omega.
        if self.config.change_omega_for_bandits or self.state == organic:
            self.omega = self.rng.normal(
                self.omega,
                self.config.sigma_omega, size=(self.config.K, 1)
            )

    # Sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli().
    def draw_click(self, recommendation):
        # Personalised CTR for every recommended product.
        ctr = self.f(matmul(self.beta, self.omega)[:, 0] + self.mu_bandit)
        click = self.rng.choice(
            [0, 1],
            p=[1 - ctr[recommendation], ctr[recommendation]]
        )
        return click

    # Sample the next organic product view.
    def sample_product_view(self):
        log_uprob = matmul(self.Gamma, self.omega)[:, 0] + self.mu_organic
        log_uprob = log_uprob - max(log_uprob)
        uprob = exp(log_uprob)
        return int(
            self.rng.choice(
                self.config.num_products,
                p=uprob / sum(uprob)
            )
        )

    def generate_beta(self, number_of_flips):
        """Create Beta by flipping Gamma, but flips are between similar items only"""

        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            return

        P, K = self.Gamma.shape
        index = list(range(P))

        prod_cov = matmul(self.Gamma, self.Gamma.T)
        prod_cov = prod_cov - diag(
            diag(prod_cov))  # We are always most correlated with ourselves so remove the diagonal.

        prod_cov_flat = prod_cov.flatten()

        already_used = dict()
        flips = 0
        pcs = prod_cov_flat.argsort()[::-1]  # Find the most correlated entries
        for ii, jj in [(int(p / P), mod(p, P)) for p in pcs]:  # Convert flat indexes to 2d indexes
            # Do flips between the most correlated entries
            # provided neither the row or col were used before.
            if not (ii in already_used or jj in already_used):
                index[ii] = jj  # Do a flip.
                index[jj] = ii
                already_used[ii] = True  # Mark as dirty.
                already_used[jj] = True
                flips += 1

                if flips == number_of_flips:
                    self.beta = self.Gamma[index, :]
                    self.mu_bandit = self.mu_organic[index]
                    return

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index]
