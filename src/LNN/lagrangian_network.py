"""Lagrangian Neural Network (LNN) module implemented in JAX.

This module implements physics-informed Lagrangian Neural Networks for 2D player
trajectory prediction based on the Euler-Lagrange equations of motion:
    d/dt (dL/d(q_dot)) - dL/dq = F_diss

Features:
- Parametric scalar Lagrangian L_theta(q, q_dot, context)
- Automatic exact differentiation using JAX grad, hessian, and jvp
- Runge-Kutta 4th-order (RK4) and Euler numerical ODE integrators
- JIT-compatible trajectory rollout using jax.lax.scan
- All mathematical operations adhere strictly to JAX (jax.numpy).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


def init_mlp_params(
    layer_sizes: List[int],
    key: jax.Array,
    scale: float = 0.05,
) -> List[Tuple[jax.Array, jax.Array]]:
    """Initializes weights and biases for a multi-layer perceptron.

    Args:
        layer_sizes: List of integers describing layer dimensions (in, hidden..., out).
        key: JAX PRNGKey.
        scale: Standard deviation scale for normal parameter initialization.

    Returns:
        List of (weight, bias) tuples.
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (in_dim, out_dim) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w = jax.random.normal(k, (in_dim, out_dim)) * scale
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        params.append((w, b))
    return params


def mlp_forward(
    params: List[Tuple[jax.Array, jax.Array]],
    x: jax.Array,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.softplus,
) -> jax.Array:
    """Computes forward pass through an MLP using JAX.

    Args:
        params: List of (weight, bias) tuples.
        x: Input tensor of shape (..., in_dim).
        activation: Activation function applied to hidden layers.

    Returns:
        Output tensor of shape (..., out_dim).
    """
    h = x
    for i, (w, b) in enumerate(params):
        h = jnp.dot(h, w) + b
        if i < len(params) - 1:
            h = activation(h)
    return h


class LagrangianNeuralNetwork:
    """Lagrangian Neural Network modeling 2D player trajectory dynamics.

    Parameterizes kinetic energy T(q, q_dot, context) and potential energy
    V(q, context) to compute the scalar Lagrangian L = T - V. Uses Euler-Lagrange
    equations to solve for generalized acceleration q_ddot and integrates the
    resulting second-order ODE.

    Attributes:
        coord_dim: Dimensionality of generalized coordinates (2 for x, y).
        context_dim: Dimensionality of conditioning context vector.
        hidden_dim: Number of units in hidden layers.
        damping: Friction coefficient for non-conservative dissipation.
        reg_epsilon: Diagonal regularization added to Hessian for inversion stability.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        context_dim: int = 16,
        hidden_dim: int = 64,
        damping: float = 0.05,
        reg_epsilon: float = 1e-4,
    ):
        """Initializes LNN configuration.

        Args:
            coord_dim: Dimensionality of generalized coordinates (default 2 for 2D position).
            context_dim: Dimension of latent player/play context vector.
            hidden_dim: Hidden dimension size for kinetic/potential networks.
            damping: Physical dissipation coefficient (turf friction / air drag).
            reg_epsilon: Diagonal damping for invertible Hessian matrix.
        """
        self.coord_dim = coord_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.damping = damping
        self.reg_epsilon = reg_epsilon

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        """Initializes model parameters for kinetic, potential, and context networks.

        Args:
            key: JAX PRNGKey.

        Returns:
            Dictionary containing model parameters.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Context encoder network: maps raw historical features to context_dim
        # Mass matrix network: takes (q, context) and outputs lower triangular L for M = L L^T + eps I
        # For 2D, lower triangular matrix has 3 unique elements (L11, L21, L22)
        n_l_elems = (self.coord_dim * (self.coord_dim + 1)) // 2

        mass_mlp = init_mlp_params(
            [self.coord_dim + self.context_dim, self.hidden_dim, self.hidden_dim, n_l_elems],
            k1,
            scale=0.05,
        )

        # Potential energy network: V(q, context) -> scalar
        potential_mlp = init_mlp_params(
            [self.coord_dim + self.context_dim, self.hidden_dim, self.hidden_dim, 1],
            k2,
            scale=0.05,
        )

        # General Lagrangian MLP (alternative or complementary residual scalar energy)
        lagrangian_mlp = init_mlp_params(
            [2 * self.coord_dim + self.context_dim, self.hidden_dim, self.hidden_dim, 1],
            k3,
            scale=0.02,
        )

        # Learnable dissipation coefficient (log-scale for positivity)
        log_damping = jnp.array([jnp.log(jnp.maximum(self.damping, 1e-4))], dtype=jnp.float32)

        return {
            "mass_mlp": mass_mlp,
            "potential_mlp": potential_mlp,
            "lagrangian_mlp": lagrangian_mlp,
            "log_damping": log_damping,
        }

    def compute_mass_matrix(
        self,
        q: jax.Array,
        context: jax.Array,
        params: Dict[str, Any],
    ) -> jax.Array:
        """Computes symmetric positive-definite mass matrix M(q, context).

        Args:
            q: Generalized coordinate vector of shape (2,).
            context: Context vector of shape (context_dim,).
            params: Model parameter dictionary.

        Returns:
            Symmetric positive-definite matrix of shape (2, 2).
        """
        inp = jnp.concatenate([q, context], axis=-1)
        l_vec = mlp_forward(params["mass_mlp"], inp)

        # For 2x2: l11, l21, l22
        l11 = jax.nn.softplus(l_vec[0]) + 0.1
        l21 = l_vec[1]
        l22 = jax.nn.softplus(l_vec[2]) + 0.1

        l_mat = jnp.array([[l11, 0.0], [l21, l22]], dtype=jnp.float32)
        mass = jnp.dot(l_mat, l_mat.T) + self.reg_epsilon * jnp.eye(self.coord_dim)
        return mass

    def potential_energy(
        self,
        q: jax.Array,
        context: jax.Array,
        params: Dict[str, Any],
    ) -> jax.Array:
        """Computes potential energy V(q, context).

        Args:
            q: Generalized coordinate vector of shape (2,).
            context: Context vector of shape (context_dim,).
            params: Model parameter dictionary.

        Returns:
            Scalar potential energy.
        """
        inp = jnp.concatenate([q, context], axis=-1)
        v = mlp_forward(params["potential_mlp"], inp)
        return jnp.squeeze(v)

    def lagrangian(
        self,
        q: jax.Array,
        q_dot: jax.Array,
        context: jax.Array,
        params: Dict[str, Any],
    ) -> jax.Array:
        """Computes total scalar Lagrangian L = T - V + L_residual.

        Args:
            q: Coordinate vector of shape (2,).
            q_dot: Velocity vector of shape (2,).
            context: Context vector of shape (context_dim,).
            params: Model parameter dictionary.

        Returns:
            Scalar Lagrangian value.
        """
        m = self.compute_mass_matrix(q, context, params)
        # Kinetic energy T = 0.5 * q_dot^T * M * q_dot
        t = 0.5 * jnp.dot(q_dot, jnp.dot(m, q_dot))
        v = self.potential_energy(q, context, params)

        # Optional residual correction
        full_inp = jnp.concatenate([q, q_dot, context], axis=-1)
        l_res = jnp.squeeze(mlp_forward(params["lagrangian_mlp"], full_inp))

        return t - v + l_res

    def euler_lagrange_derivatives(
        self,
        q: jax.Array,
        q_dot: jax.Array,
        context: jax.Array,
        params: Dict[str, Any],
    ) -> Tuple[jax.Array, jax.Array]:
        """Calculates state derivatives (q_dot, q_ddot) using Euler-Lagrange equations.

        From d/dt (dL/dq_dot) - dL/dq = -gamma * q_dot:
        q_ddot = (d^2 L / dq_dot^2)^(-1) * [ dL/dq - (d^2 L / dq dq_dot) * q_dot - gamma * q_dot ]

        Args:
            q: Position coordinate vector of shape (2,).
            q_dot: Velocity vector of shape (2,).
            context: Context vector of shape (context_dim,).
            params: Model parameter dictionary.

        Returns:
            Tuple of (q_dot, q_ddot) each of shape (2,).
        """
        # Gradient with respect to coordinates: dL / dq
        grad_q = jax.grad(self.lagrangian, argnums=0)(q, q_dot, context, params)

        # Hessian with respect to velocity: d^2 L / dq_dot^2
        hess_qdot = jax.hessian(self.lagrangian, argnums=1)(q, q_dot, context, params)
        # Add diagonal damping for unconditional numerical stability
        hess_qdot_reg = hess_qdot + self.reg_epsilon * jnp.eye(self.coord_dim)

        # Mixed derivative contraction: (d^2 L / dq dq_dot) * q_dot via directional derivative
        def grad_qdot_fn(coord_var: jax.Array) -> jax.Array:
            return jax.grad(self.lagrangian, argnums=1)(coord_var, q_dot, context, params)

        _, mixed_term = jax.jvp(grad_qdot_fn, (q,), (q_dot,))

        # Dissipative force
        damping_coeff = jnp.exp(params["log_damping"][0])
        dissipative_force = -damping_coeff * q_dot

        # Right-hand side vector
        rhs = grad_q - mixed_term + dissipative_force

        # Solve linear system for q_ddot
        q_ddot = jnp.linalg.solve(hess_qdot_reg, rhs)
        return q_dot, q_ddot

    def rk4_step(
        self,
        state: Tuple[jax.Array, jax.Array],
        context: jax.Array,
        params: Dict[str, Any],
        dt: float = 0.1,
    ) -> Tuple[jax.Array, jax.Array]:
        """Performs a single 4th-order Runge-Kutta numerical integration step.

        Args:
            state: Tuple of current (q, q_dot) vectors.
            context: Latent context vector.
            params: Model parameters.
            dt: Time delta between frames (default 0.1s for 10Hz tracking).

        Returns:
            Updated (next_q, next_q_dot) state.
        """
        q, q_dot = state

        # k1
        dq1, dv1 = self.euler_lagrange_derivatives(q, q_dot, context, params)

        # k2
        q2 = q + 0.5 * dt * dq1
        v2 = q_dot + 0.5 * dt * dv1
        dq2, dv2 = self.euler_lagrange_derivatives(q2, v2, context, params)

        # k3
        q3 = q + 0.5 * dt * dq2
        v3 = q_dot + 0.5 * dt * dv2
        dq3, dv3 = self.euler_lagrange_derivatives(q3, v3, context, params)

        # k4
        q4 = q + dt * dq3
        v4 = q_dot + dt * dv3
        dq4, dv4 = self.euler_lagrange_derivatives(q4, v4, context, params)

        # Combine RK4 stages
        next_q = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
        next_q_dot = q_dot + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)

        return next_q, next_q_dot

    def rollout_trajectory(
        self,
        q0: jax.Array,
        q_dot0: jax.Array,
        context: jax.Array,
        params: Dict[str, Any],
        horizon: int,
        dt: float = 0.1,
    ) -> jax.Array:
        """Rolls out predicted future coordinates forward in time using jax.lax.scan.

        Args:
            q0: Initial position vector of shape (2,).
            q_dot0: Initial velocity vector of shape (2,).
            context: Context vector of shape (context_dim,).
            params: Model parameter dictionary.
            horizon: Number of future timesteps to simulate.
            dt: Integration time step (seconds).

        Returns:
            Predicted trajectory positions of shape (horizon, 2).
        """
        def scan_fn(carry_state: Tuple[jax.Array, jax.Array], _step: Any):
            next_state = self.rk4_step(carry_state, context, params, dt=dt)
            return next_state, next_state[0]

        _, predicted_qs = jax.lax.scan(scan_fn, (q0, q_dot0), None, length=horizon)
        return predicted_qs


class ContextEncoder:
    """Encodes historical tracking observations into a conditioning context vector.

    Attributes:
        input_dim: Number of input features per frame.
        context_dim: Dimension of resulting context vector.
        hidden_dim: Number of hidden units in projection MLP.
    """

    def __init__(self, input_dim: int, context_dim: int = 16, hidden_dim: int = 64):
        """Initializes context encoder network structure.

        Args:
            input_dim: Number of features in input tracking sequence.
            context_dim: Output context representation dimension.
            hidden_dim: Hidden dimension size.
        """
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        """Initializes parameters for the context encoder.

        Args:
            key: JAX PRNGKey.

        Returns:
            Dictionary containing MLP parameters.
        """
        mlp_params = init_mlp_params(
            [self.input_dim, self.hidden_dim, self.context_dim],
            key,
            scale=0.05,
        )
        return {"encoder_mlp": mlp_params}

    def encode(self, sequence: jax.Array, params: Dict[str, Any]) -> jax.Array:
        """Compresses historical sequence into a single context vector.

        Computes mean-pooled MLP representation over valid timesteps.

        Args:
            sequence: Historical array of shape (seq_len, input_dim).
            params: Encoder parameter dictionary.

        Returns:
            Context vector of shape (context_dim,).
        """
        frame_embeddings = mlp_forward(params["encoder_mlp"], sequence)
        return jnp.mean(frame_embeddings, axis=0)


class FullLNNModel:
    """End-to-end model coupling ContextEncoder and LagrangianNeuralNetwork.

    Attributes:
        context_encoder: ContextEncoder instance.
        lnn: LagrangianNeuralNetwork instance.
    """

    def __init__(
        self,
        input_dim: int,
        coord_dim: int = 2,
        context_dim: int = 16,
        hidden_dim: int = 64,
        damping: float = 0.05,
    ):
        """Initializes the full model.

        Args:
            input_dim: Number of input features per frame.
            coord_dim: Dimension of target coordinates (default 2).
            context_dim: Latent context dimension.
            hidden_dim: Hidden layer size for networks.
            damping: Friction damping coefficient.
        """
        self.context_encoder = ContextEncoder(
            input_dim=input_dim, context_dim=context_dim, hidden_dim=hidden_dim
        )
        self.lnn = LagrangianNeuralNetwork(
            coord_dim=coord_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            damping=damping,
        )

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        """Initializes all model parameters.

        Args:
            key: JAX PRNGKey.

        Returns:
            Unified parameter dictionary.
        """
        k1, k2 = jax.random.split(key)
        return {
            "encoder": self.context_encoder.init_params(k1),
            "lnn": self.lnn.init_params(k2),
        }

    def predict_single(
        self,
        input_seq: jax.Array,
        horizon: int,
        params: Dict[str, Any],
        x_idx: int = 0,
        y_idx: int = 1,
        dt: float = 0.1,
    ) -> jax.Array:
        """Predicts future trajectory for a single sequence.

        Args:
            input_seq: Array of shape (seq_len, input_dim).
            horizon: Future prediction horizon length.
            params: Unified model parameters.
            x_idx: Index of x-coordinate in input_seq features.
            y_idx: Index of y-coordinate in input_seq features.
            dt: Time step (seconds).

        Returns:
            Predicted coordinates of shape (horizon, 2).
        """
        context = self.context_encoder.encode(input_seq, params["encoder"])

        # Extract initial state from last observed frames
        last_frame = input_seq[-1]
        q0 = jnp.array([last_frame[x_idx], last_frame[y_idx]], dtype=jnp.float32)

        # Velocity estimate from last two frames or zero if single frame
        num_frames = input_seq.shape[0]
        q_prev = (
            jnp.array([input_seq[-2, x_idx], input_seq[-2, y_idx]], dtype=jnp.float32)
            if num_frames > 1
            else q0
        )
        q_dot0 = (q0 - q_prev) / jnp.maximum(dt, 1e-4)

        return self.lnn.rollout_trajectory(
            q0=q0,
            q_dot0=q_dot0,
            context=context,
            params=params["lnn"],
            horizon=horizon,
            dt=dt,
        )

    def predict_batch(
        self,
        batch_input: jax.Array,
        horizon: int,
        params: Dict[str, Any],
        x_idx: int = 0,
        y_idx: int = 1,
        dt: float = 0.1,
    ) -> jax.Array:
        """Vectorized trajectory rollout across a batch of sequences.

        Args:
            batch_input: Tensor of shape (batch_size, seq_len, input_dim).
            horizon: Number of future timesteps.
            params: Model parameters.
            x_idx: Column index for x position.
            y_idx: Column index for y position.
            dt: Time step between frames.

        Returns:
            Tensor of shape (batch_size, horizon, 2).
        """
        predict_fn = lambda seq: self.predict_single(
            seq, horizon, params, x_idx=x_idx, y_idx=y_idx, dt=dt
        )
        return jax.vmap(predict_fn)(batch_input)
