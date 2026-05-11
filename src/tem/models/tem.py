import torch 
from torch import nn

from tem.config import TEMConfig
from tem.models.activations import EPS, phi_g, phi_p, positive_softplus
from tem.models.attractor import Attractor
from tem.models.tensor_ops import l2_normalize, phase_sum, repeat_sensory, sense_sum, tile_phase
from tem.types import TEMSequenceOutput, TEMState


class TEM(nn.Module):
    """
    Minimal end-to-end Tolman-Eichenbaum Machine implementation.

    This class intentionally keeps the whole forward computation in one file.
    For research code, this makes it easier to compare the implementation with
    the mathematical computation flow.

    Main design choice:
        - Keep small reusable tensor functions in tensor_ops.py.
        - Keep the attractor retrieval module in attractor.py.
        - Keep the complete TEM step logic here, in a clearly divided forward loop.

    Tensor conventions:
        B     = batch size
        T     = sequence length
        N_x   = number of sensory observation categories
        N_a   = number of action / relation categories
        F     = number of modules
        C     = compressed sensory dimension
        G_f   = grid dimension of module f
        Phi_f = phase dimension of module f
        P_f   = Phi_f * C
    """

    def __init__(self, config: TEMConfig) -> None:
        super().__init__()

        self.config = config

        data_config = config.data
        model_config = config.model

        self.num_observations = data_config.num_observations
        self.num_actions = data_config.num_actions

        self.num_modules = model_config.num_modules
        self.compressed_x_dim = model_config.compressed_x_dim

        self.g_dims = model_config.g_dims
        self.phase_dims = model_config.phase_dims
        self.p_dims = [
            phase_dim * self.compressed_x_dim
            for phase_dim in self.phase_dims
        ]

        self.sensory_filter_alphas = model_config.sensory_filter_alphas
        self.sensory_scales = model_config.sensory_scales
        self.attractor_kappas = model_config.attractor_kappas

        self.memory_eta_gen = model_config.memory_eta_gen
        self.memory_eta_inf = model_config.memory_eta_inf
        self.memory_lambda_gen = model_config.memory_lambda_gen
        self.memory_lambda_inf = model_config.memory_lambda_inf
        self.memory_clip = model_config.memory_clip

        # ------------------------------------------------------------
        # 1. Sensory encoder
        # ------------------------------------------------------------
        #
        # Mathematical shape in the computation document:
        #     W_xc: [N_x, C]
        #     b_xc: [C]
        #
        # PyTorch nn.Linear stores weight as [C, N_x], but computes:
        #     x @ W_xc.T + b_xc
        #
        # Input:
        #     x_t: [B, N_x]
        #
        # Output:
        #     x_c_t: [B, C]
        self.sensory_encoder = nn.Linear(
            self.num_observations,
            self.compressed_x_dim,
        )

        # ------------------------------------------------------------
        # 2. Structural transition parameters
        # ------------------------------------------------------------
        #
        # For each module f:
        #
        #     action a_t -> transition matrix D_t[f]
        #     action a_t -> path uncertainty sigma_path_t[f]
        #
        # transition_d[f]:
        #     input:  [B, N_a]
        #     output: [B, G_f * G_f]
        #
        # transition_sigma[f]:
        #     input:  [B, N_a]
        #     output: [B, G_f]
        self.transition_d = nn.ModuleList()
        self.transition_sigma = nn.ModuleList()

        for g_dim in self.g_dims:
            self.transition_d.append(
                nn.Linear(self.num_actions, g_dim * g_dim)
            )
            self.transition_sigma.append(
                nn.Linear(self.num_actions, g_dim)
            )

        # ------------------------------------------------------------
        # 3. Memory-to-grid inference parameters
        # ------------------------------------------------------------
        #
        # For each module f:
        #
        #     p_x_t[f] -> u_x_t[f] -> mu_mem_t[f], sigma_mem_t[f]
        #
        # memory_mu[f]:
        #     input:  [B, Phi_f]
        #     output: [B, G_f]
        #
        # memory_sigma[f]:
        #     input:  [B, Phi_f]
        #     output: [B, G_f]
        self.memory_mu = nn.ModuleList()
        self.memory_sigma = nn.ModuleList()

        for phase_dim, g_dim in zip(self.phase_dims, self.g_dims):
            self.memory_mu.append(
                nn.Linear(phase_dim, g_dim)
            )
            self.memory_sigma.append(
                nn.Linear(phase_dim, g_dim)
            )

        # ------------------------------------------------------------
        # 4. Grid-to-phase projection
        # ------------------------------------------------------------
        #
        # For each module f:
        #
        #     g_t[f] -> r_t[f]
        #
        # grid_to_phase[f]:
        #     input:  [B, G_f]
        #     output: [B, Phi_f]
        self.grid_to_phase = nn.ModuleList()

        for g_dim, phase_dim in zip(self.g_dims, self.phase_dims):
            self.grid_to_phase.append(
                nn.Linear(g_dim, phase_dim)
            )

        # ------------------------------------------------------------
        # 5. Shared decoder
        # ------------------------------------------------------------
        #
        # Each module produces one compressed sensory vector [B, C].
        # All modules are concatenated into [B, F * C].
        #
        # decoder:
        #     input:  [B, F * C]
        #     output: [B, N_x]
        self.decoder = nn.Linear(
            self.num_modules * self.compressed_x_dim,
            self.num_observations,
        )

        # ------------------------------------------------------------
        # 6. Attractor retrieval
        # ------------------------------------------------------------
        #
        # The same attractor object is reused for all modules.
        # Module-specific behavior comes from:
        #     - P_f
        #     - memory[f]
        #     - kappa[f]
        self.attractor = Attractor(model_config.attractor_iterations)


    

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TEMState:
        """
        Create a zero initial recurrent state.

        For each module f:

            g[f]:          [B, G_f]
            x_s[f]:        [B, C]
            memory_gen[f]: [B, P_f, P_f]
            memory_inf[f]: [B, P_f, P_f]
        """
        g = []
        x_s = []
        memory_gen = []
        memory_inf = []

        for g_dim, p_dim in zip(self.g_dims, self.p_dims):
            g.append(
                torch.zeros(batch_size, g_dim, device=device, dtype=dtype)
            )
            x_s.append(
                torch.zeros(
                    batch_size,
                    self.compressed_x_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            memory_gen.append(
                torch.zeros(batch_size, p_dim, p_dim, device=device, dtype=dtype)
            )
            memory_inf.append(
                torch.zeros(batch_size, p_dim, p_dim, device=device, dtype=dtype)
            )

        return TEMState(
            g=g,
            x_s=x_s,
            memory_gen=memory_gen,
            memory_inf=memory_inf,
        )

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    def _check_inputs(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        visited: torch.Tensor,
    ) -> None:
        """
        Check external input shapes before running the recurrent loop.

        Expected shapes:
            x:       [B, T, N_x]
            a:       [B, T, N_a]
            visited: [B, T]
        """
        if x.dim() != 3:
            raise ValueError(
                f"x must have shape [B, T, N_x], but got {tuple(x.shape)}."
            )

        if a.dim() != 3:
            raise ValueError(
                f"a must have shape [B, T, N_a], but got {tuple(a.shape)}."
            )

        if visited.dim() != 2:
            raise ValueError(
                f"visited must have shape [B, T], but got {tuple(visited.shape)}."
            )

        batch_size, sequence_length, num_observations = x.shape

        if a.shape[0] != batch_size or a.shape[1] != sequence_length:
            raise ValueError(
                "a must have the same batch and time dimensions as x. "
                f"x shape: {tuple(x.shape)}, a shape: {tuple(a.shape)}."
            )

        if visited.shape != (batch_size, sequence_length):
            raise ValueError(
                "visited must have shape [B, T] matching x. "
                f"x shape: {tuple(x.shape)}, visited shape: {tuple(visited.shape)}."
            )

        if num_observations != self.num_observations:
            raise ValueError(
                f"x last dimension must be {self.num_observations}, "
                f"but got {num_observations}."
            )

        if a.shape[-1] != self.num_actions:
            raise ValueError(
                f"a last dimension must be {self.num_actions}, "
                f"but got {a.shape[-1]}."
            )

        if sequence_length <= 0:
            raise ValueError("sequence length must be positive.")

        if not x.is_floating_point():
            raise ValueError("x must be a floating point tensor.")

        if not a.is_floating_point():
            raise ValueError("a must be a floating point tensor.")

        model_device = next(self.parameters()).device

        if x.device != model_device:
            raise ValueError(
                f"x is on device {x.device}, but model is on device {model_device}. "
                "Move the model or inputs so they are on the same device."
            )

        if a.device != x.device or visited.device != x.device:
            raise ValueError(
                "x, a, and visited must be on the same device."
            )

    def _check_state(self, state: TEMState, batch_size: int, device: torch.device,) -> None:
        """
        Check recurrent state shapes.

        This is mainly useful when passing a custom initial_state.
        """
        state_lists = {
            "g": state.g,
            "x_s": state.x_s,
            "memory_gen": state.memory_gen,
            "memory_inf": state.memory_inf,
        }

        for name, values in state_lists.items():
            if len(values) != self.num_modules:
                raise ValueError(
                    f"state.{name} must have length {self.num_modules}, "
                    f"but got {len(values)}."
                )

        for f in range(self.num_modules):
            g_dim = self.g_dims[f]
            p_dim = self.p_dims[f]

            expected_g_shape = (batch_size, g_dim)
            expected_x_s_shape = (batch_size, self.compressed_x_dim)
            expected_memory_shape = (batch_size, p_dim, p_dim)

            if tuple(state.g[f].shape) != expected_g_shape:
                raise ValueError(
                    f"state.g[{f}] must have shape {expected_g_shape}, "
                    f"but got {tuple(state.g[f].shape)}."
                )

            if tuple(state.x_s[f].shape) != expected_x_s_shape:
                raise ValueError(
                    f"state.x_s[{f}] must have shape {expected_x_s_shape}, "
                    f"but got {tuple(state.x_s[f].shape)}."
                )

            if tuple(state.memory_gen[f].shape) != expected_memory_shape:
                raise ValueError(
                    f"state.memory_gen[{f}] must have shape {expected_memory_shape}, "
                    f"but got {tuple(state.memory_gen[f].shape)}."
                )

            if tuple(state.memory_inf[f].shape) != expected_memory_shape:
                raise ValueError(
                    f"state.memory_inf[{f}] must have shape {expected_memory_shape}, "
                    f"but got {tuple(state.memory_inf[f].shape)}."
                )

            tensors = [
                state.g[f],
                state.x_s[f],
                state.memory_gen[f],
                state.memory_inf[f],
            ]

            for tensor in tensors:
                if tensor.device != device:
                    raise ValueError(
                        f"All state tensors must be on device {device}, "
                        f"but found tensor on device {tensor.device}."
                    )







    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, a: torch.Tensor, visited: torch.Tensor, initial_state=None) -> TEMSequenceOutput:
        """
        Run the full TEM sequence computation.

        Inputs:
            x:
                Sensory observations.
                Shape: [B, T, N_x]

            a:
                Actions / relations.
                Shape: [B, T, N_a]

            visited:
                Memory update mask.
                Shape: [B, T]

                visited[b, t] = 1 means:
                    update memory for batch item b at time t.

                visited[b, t] = 0 means:
                    keep previous memory unchanged for batch item b at time t.

            initial_state:
                Optional TEMState.
                If None, zero state is used.

        Output:
            TEMSequenceOutput with all major intermediate tensors stacked
            along the time dimension.
        """
        self._check_inputs(x, a, visited)

        batch_size, sequence_length, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Convert visited to the same dtype as x.
        # This supports both float masks and bool masks.
        visited = visited.to(dtype=dtype)

        if initial_state is None:
            state = self.initial_state(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            self._check_state(
                state=initial_state,
                batch_size=batch_size,
                device=device,
            )
            state = initial_state

        # ------------------------------------------------------------
        # Per-time-step output containers
        # ------------------------------------------------------------

        x_p_logits_steps = []
        x_g_logits_steps = []
        x_gt_logits_steps = []

        x_p_steps = []
        x_g_steps = []
        x_gt_steps = []

        # Each module gets its own list of time-step tensors.
        g_steps = [[] for _ in range(self.num_modules)]
        g_gen_steps = [[] for _ in range(self.num_modules)]

        p_steps = [[] for _ in range(self.num_modules)]
        p_x_steps = [[] for _ in range(self.num_modules)]
        p_g_steps = [[] for _ in range(self.num_modules)]
        p_gt_steps = [[] for _ in range(self.num_modules)]

        x_s_steps = [[] for _ in range(self.num_modules)]

        # ============================================================
        # Main recurrent sequence loop
        # ============================================================

        for t in range(sequence_length):
            # --------------------------------------------------------
            # 1. Read current inputs
            # --------------------------------------------------------
            #
            # x_t:
            #     [B, N_x]
            #
            # a_t:
            #     [B, N_a]
            #
            # visited_memory_mask:
            #     [B, 1, 1]
            #
            # The memory mask is shaped for broadcasting over
            # [B, P_f, P_f].
            x_t = x[:, t, :]
            a_t = a[:, t, :]

            visited_t = visited[:, t]
            visited_memory_mask = visited_t.reshape(batch_size, 1, 1)

            # --------------------------------------------------------
            # 2. Sensory encoding
            # --------------------------------------------------------
            #
            # x_c_t:
            #     [B, C]
            #
            # The same compressed sensory vector is used by all modules.
            x_c_t = phi_p(self.sensory_encoder(x_t))

            # These lists collect module-specific tensors for this
            # single time step.
            g_next = []
            x_s_next = []
            memory_gen_next = []
            memory_inf_next = []

            z_p_parts = []
            z_g_parts = []
            z_gt_parts = []

            for f in range(self.num_modules):
                g_dim = self.g_dims[f]
                phase_dim = self.phase_dims[f]
                p_dim = self.p_dims[f]

                alpha = self.sensory_filter_alphas[f]
                gamma = self.sensory_scales[f]
                kappa = self.attractor_kappas[f]

                g_prev_f = state.g[f]
                x_s_prev_f = state.x_s[f]
                memory_gen_prev_f = state.memory_gen[f]
                memory_inf_prev_f = state.memory_inf[f]

                # ----------------------------------------------------
                # 2.1 Module-specific filtered sensory trace
                # ----------------------------------------------------
                #
                # x_s_t_f:
                #     [B, C]
                #
                # This is an exponential moving average of compressed
                # sensory input.
                x_s_t_f = (
                    (1.0 - alpha) * x_s_prev_f
                    + alpha * x_c_t
                )

                # Center the sensory trace by subtracting its mean.
                #
                # x_s_centered_t_f:
                #     [B, C]
                #
                # ReLU keeps only positive deviations from the mean.
                x_s_mean_t_f = x_s_t_f.mean(dim=-1, keepdim=True)
                x_s_centered_t_f = torch.relu(x_s_t_f - x_s_mean_t_f)

                # Normalize sensory vector along C.
                #
                # x_s_norm_t_f:
                #     [B, C]
                x_s_norm_t_f = l2_normalize(x_s_centered_t_f)

                # Tile sensory vector across phase dimension.
                #
                # x_2p_t_f:
                #     [B, P_f]
                #
                # P_f = Phi_f * C.
                x_2p_t_f = gamma * tile_phase(
                    x_s_norm_t_f,
                    num_phases=phase_dim,
                )

                # ----------------------------------------------------
                # 3. Structural transition
                # ----------------------------------------------------
                #
                # d_t_f:
                #     [B, G_f * G_f]
                #
                # D_t_f:
                #     [B, G_f, G_f]
                d_t_f = self.transition_d[f](a_t)
                D_t_f = d_t_f.reshape(batch_size, g_dim, g_dim)

                # Apply the action-conditioned transition matrix to
                # the previous grid state.
                #
                # delta_g_t_f:
                #     [B, G_f]
                delta_g_t_f = torch.bmm(
                    g_prev_f.unsqueeze(1),
                    D_t_f,
                ).squeeze(1)

                # Path-integrated grid estimate.
                #
                # mu_path_t_f:
                #     [B, G_f]
                mu_path_t_f = phi_g(g_prev_f + delta_g_t_f)

                # Path uncertainty.
                #
                # sigma_path_t_f:
                #     [B, G_f]
                sigma_path_t_f = positive_softplus(
                    self.transition_sigma[f](a_t)
                )

                # g_gen_t_f is the transition-only grid prediction.
                #
                # g_gen_t_f:
                #     [B, G_f]
                g_gen_t_f = mu_path_t_f

                # ----------------------------------------------------
                # 4. Inference memory retrieval from sensory cue
                # ----------------------------------------------------
                #
                # p_x_t_f:
                #     [B, P_f]
                #
                # Query is sensory-bound p-space input x_2p_t_f.
                # Memory is inference memory from the previous step.
                p_x_t_f = self.attractor(
                    query=x_2p_t_f,
                    memory=memory_inf_prev_f,
                    kappa=kappa,
                )

                # Collapse sensory dimension and keep phase dimension.
                #
                # u_x_t_f:
                #     [B, Phi_f]
                u_x_t_f = sense_sum(
                    p_x_t_f,
                    num_phases=phase_dim,
                    compressed_x_dim=self.compressed_x_dim,
                )

                # Memory-based grid estimate.
                #
                # mu_mem_t_f:
                #     [B, G_f]
                mu_mem_t_f = phi_g(
                    self.memory_mu[f](u_x_t_f)
                )

                # Memory uncertainty.
                #
                # sigma_mem_t_f:
                #     [B, G_f]
                sigma_mem_t_f = positive_softplus(
                    self.memory_sigma[f](u_x_t_f)
                )

                # ----------------------------------------------------
                # 5. Combine path estimate and memory estimate
                # ----------------------------------------------------
                #
                # This is precision-weighted averaging.
                #
                # Lower variance means higher precision.
                var_path_t_f = sigma_path_t_f ** 2
                var_mem_t_f = sigma_mem_t_f ** 2

                precision_path_t_f = 1.0 / (var_path_t_f + EPS)
                precision_mem_t_f = 1.0 / (var_mem_t_f + EPS)

                precision_total_t_f = (
                    precision_path_t_f
                    + precision_mem_t_f
                )

                mu_g_t_f = (
                    mu_path_t_f * precision_path_t_f
                    + mu_mem_t_f * precision_mem_t_f
                ) / (precision_total_t_f + EPS)

                # Final inferred grid state.
                #
                # g_t_f:
                #     [B, G_f]
                g_t_f = phi_g(mu_g_t_f)

                # ----------------------------------------------------
                # 6. Project grid state to p-space
                # ----------------------------------------------------
                #
                # r_t_f:
                #     [B, Phi_f]
                #
                # g_2p_t_f:
                #     [B, P_f]
                r_t_f = phi_g(
                    self.grid_to_phase[f](g_t_f)
                )
                g_2p_t_f = repeat_sensory(
                    r_t_f,
                    compressed_x_dim=self.compressed_x_dim,
                )

                # Also project transition-only grid state.
                #
                # r_gen_t_f:
                #     [B, Phi_f]
                #
                # g_gen_2p_t_f:
                #     [B, P_f]
                r_gen_t_f = phi_g(
                    self.grid_to_phase[f](g_gen_t_f)
                )
                g_gen_2p_t_f = repeat_sensory(
                    r_gen_t_f,
                    compressed_x_dim=self.compressed_x_dim,
                )

                # ----------------------------------------------------
                # 7. Infer bound place representation
                # ----------------------------------------------------
                #
                # p_t_f:
                #     [B, P_f]
                #
                # This binds grid-derived phase structure with
                # sensory-derived content.
                p_t_f = phi_p(g_2p_t_f * x_2p_t_f)

                # ----------------------------------------------------
                # 8. Generative memory retrieval
                # ----------------------------------------------------
                #
                # p_g_t_f:
                #     [B, P_f]
                #
                # Query uses inferred grid state.
                p_g_t_f = self.attractor(
                    query=g_2p_t_f,
                    memory=memory_gen_prev_f,
                    kappa=kappa,
                )

                # ----------------------------------------------------
                # 9. Transition-only generative retrieval
                # ----------------------------------------------------
                #
                # p_gt_t_f:
                #     [B, P_f]
                #
                # Query uses transition-only grid state.
                p_gt_t_f = self.attractor(
                    query=g_gen_2p_t_f,
                    memory=memory_gen_prev_f,
                    kappa=kappa,
                )

                # ----------------------------------------------------
                # 10. Prepare module contribution for decoding
                # ----------------------------------------------------
                #
                # Each p-space vector is collapsed over phase dimension.
                #
                # z_*_t_f:
                #     [B, C]
                z_p_t_f = phase_sum(
                    p_t_f,
                    num_phases=phase_dim,
                    compressed_x_dim=self.compressed_x_dim,
                )
                z_g_t_f = phase_sum(
                    p_g_t_f,
                    num_phases=phase_dim,
                    compressed_x_dim=self.compressed_x_dim,
                )
                z_gt_t_f = phase_sum(
                    p_gt_t_f,
                    num_phases=phase_dim,
                    compressed_x_dim=self.compressed_x_dim,
                )

                z_p_parts.append(z_p_t_f)
                z_g_parts.append(z_g_t_f)
                z_gt_parts.append(z_gt_t_f)

                # ----------------------------------------------------
                # 11. Hebbian memory update
                # ----------------------------------------------------
                #
                # Generative memory update:
                #
                #     a_gen = p - p_g
                #     b_gen = p + p_g
                #     delta_memory_gen = outer(b_gen, a_gen)
                #
                # Shape:
                #     delta_memory_gen_t_f: [B, P_f, P_f]
                a_gen_t_f = p_t_f - p_g_t_f
                b_gen_t_f = p_t_f + p_g_t_f

                delta_memory_gen_t_f = (
                    b_gen_t_f[:, :, None]
                    * a_gen_t_f[:, None, :]
                )

                memory_gen_candidate_t_f = torch.clamp(
                    self.memory_lambda_gen * memory_gen_prev_f
                    + self.memory_eta_gen * delta_memory_gen_t_f,
                    min=-self.memory_clip,
                    max=self.memory_clip,
                )

                memory_gen_t_f = (
                    visited_memory_mask * memory_gen_candidate_t_f
                    + (1.0 - visited_memory_mask) * memory_gen_prev_f
                )

                # Inference memory update:
                #
                #     a_inf = p - p_x
                #     b_inf = p + p_x
                #     delta_memory_inf = outer(b_inf, a_inf)
                #
                # Shape:
                #     delta_memory_inf_t_f: [B, P_f, P_f]
                a_inf_t_f = p_t_f - p_x_t_f
                b_inf_t_f = p_t_f + p_x_t_f

                delta_memory_inf_t_f = (
                    b_inf_t_f[:, :, None]
                    * a_inf_t_f[:, None, :]
                )

                memory_inf_candidate_t_f = torch.clamp(
                    self.memory_lambda_inf * memory_inf_prev_f
                    + self.memory_eta_inf * delta_memory_inf_t_f,
                    min=-self.memory_clip,
                    max=self.memory_clip,
                )

                memory_inf_t_f = (
                    visited_memory_mask * memory_inf_candidate_t_f
                    + (1.0 - visited_memory_mask) * memory_inf_prev_f
                )

                # ----------------------------------------------------
                # 12. Save module outputs for this time step
                # ----------------------------------------------------
                g_next.append(g_t_f)
                x_s_next.append(x_s_t_f)
                memory_gen_next.append(memory_gen_t_f)
                memory_inf_next.append(memory_inf_t_f)

                g_steps[f].append(g_t_f)
                g_gen_steps[f].append(g_gen_t_f)

                p_steps[f].append(p_t_f)
                p_x_steps[f].append(p_x_t_f)
                p_g_steps[f].append(p_g_t_f)
                p_gt_steps[f].append(p_gt_t_f)

                x_s_steps[f].append(x_s_t_f)

                # p_dim is not used directly below, but keeping this
                # assertion here makes module shape errors easier to catch
                # during early research implementation.
                assert p_t_f.shape[-1] == p_dim

            # --------------------------------------------------------
            # 13. Decode predictions from all modules
            # --------------------------------------------------------
            #
            # Concatenate module-wise compressed sensory vectors.
            #
            # z_p_t:
            #     [B, F * C]
            z_p_t = torch.cat(z_p_parts, dim=-1)
            z_g_t = torch.cat(z_g_parts, dim=-1)
            z_gt_t = torch.cat(z_gt_parts, dim=-1)

            # Decode to observation logits.
            #
            # x_*_logits_t:
            #     [B, N_x]
            x_p_logits_t = self.decoder(z_p_t)
            x_g_logits_t = self.decoder(z_g_t)
            x_gt_logits_t = self.decoder(z_gt_t)

            # Convert logits to probabilities.
            #
            # x_*_t:
            #     [B, N_x]
            x_p_t = torch.softmax(x_p_logits_t, dim=-1)
            x_g_t = torch.softmax(x_g_logits_t, dim=-1)
            x_gt_t = torch.softmax(x_gt_logits_t, dim=-1)

            x_p_logits_steps.append(x_p_logits_t)
            x_g_logits_steps.append(x_g_logits_t)
            x_gt_logits_steps.append(x_gt_logits_t)

            x_p_steps.append(x_p_t)
            x_g_steps.append(x_g_t)
            x_gt_steps.append(x_gt_t)

            # --------------------------------------------------------
            # 14. Carry recurrent state to next time step
            # --------------------------------------------------------
            state = TEMState(
                g=g_next,
                x_s=x_s_next,
                memory_gen=memory_gen_next,
                memory_inf=memory_inf_next,
            )

        # ============================================================
        # Stack all per-step outputs along time dimension
        # ============================================================

        x_p_logits = torch.stack(x_p_logits_steps, dim=1)
        x_g_logits = torch.stack(x_g_logits_steps, dim=1)
        x_gt_logits = torch.stack(x_gt_logits_steps, dim=1)

        x_p = torch.stack(x_p_steps, dim=1)
        x_g = torch.stack(x_g_steps, dim=1)
        x_gt = torch.stack(x_gt_steps, dim=1)

        g = [
            torch.stack(g_steps[f], dim=1)
            for f in range(self.num_modules)
        ]
        g_gen = [
            torch.stack(g_gen_steps[f], dim=1)
            for f in range(self.num_modules)
        ]

        p = [
            torch.stack(p_steps[f], dim=1)
            for f in range(self.num_modules)
        ]
        p_x = [
            torch.stack(p_x_steps[f], dim=1)
            for f in range(self.num_modules)
        ]
        p_g = [
            torch.stack(p_g_steps[f], dim=1)
            for f in range(self.num_modules)
        ]
        p_gt = [
            torch.stack(p_gt_steps[f], dim=1)
            for f in range(self.num_modules)
        ]

        x_s = [
            torch.stack(x_s_steps[f], dim=1)
            for f in range(self.num_modules)
        ]

        return TEMSequenceOutput(
            x_p_logits=x_p_logits,
            x_g_logits=x_g_logits,
            x_gt_logits=x_gt_logits,
            x_p=x_p,
            x_g=x_g,
            x_gt=x_gt,
            g=g,
            g_gen=g_gen,
            p=p,
            p_x=p_x,
            p_g=p_g,
            p_gt=p_gt,
            x_s=x_s,
            final_state=state,
        )