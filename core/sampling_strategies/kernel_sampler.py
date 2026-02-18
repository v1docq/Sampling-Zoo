import math

from dataclasses import dataclass
from typing import Optional, List, Dict, Literal

import torch


# -------------------------
# Utilities: matching stats
# -------------------------

@torch.no_grad()
def match_count_matrix(
    X_cat: torch.Tensor,  # [N, F] long
    A_cat: torch.Tensor,  # [m, F] long
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Returns t = number of matching fields between each x_i and each anchor a_j.
    Output: t [N, m] int16/int32
    Memory: O(chunk_size * m).
    """
    assert X_cat.dtype == torch.long and A_cat.dtype == torch.long
    N, F = X_cat.shape
    m, F2 = A_cat.shape
    assert F == F2

    device = X_cat.device
    t_all = torch.empty((N, m), device=device, dtype=torch.int16 if F <= 32767 else torch.int32)

    # Accumulate matches per field without building [N, m, F]
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        Xb = X_cat[start:end]  # [B, F]
        # t_b [B, m]
        t_b = torch.zeros((Xb.shape[0], m), device=device, dtype=torch.int16 if F <= 32767 else torch.int32)
        for f in range(F):
            # [B, 1] == [1, m] => [B, m]
            t_b += (Xb[:, f].unsqueeze(1) == A_cat[:, f].unsqueeze(0)).to(t_b.dtype)
        t_all[start:end] = t_b
    return t_all


@torch.no_grad()
def weighted_logprod_mdnf_kernel_XA(
    X_cat: torch.Tensor,      # [N, F] long
    A_cat: torch.Tensor,      # [m, F] long
    w_f: torch.Tensor,        # [F] float
    sigma: float = 1.0,
    chunk_size: int = 65536,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Field-aware σ-mDNF kernel between samples and anchors:
      K(x,a) = prod_{f: x_f == a_f} (1 + sigma*w_f) - 1
    Computed via logs:
      log(1 + K) = sum_{f} 1[x_f==a_f] * log(1 + sigma*w_f)
    Output: K_XA [N, m] float32/float16 depending on inputs.
    """
    assert X_cat.dtype == torch.long and A_cat.dtype == torch.long
    assert w_f.ndim == 1 and w_f.shape[0] == X_cat.shape[1]
    device = X_cat.device
    w_f = w_f.to(device=device, dtype=torch.float32)

    # log factors per field
    log_f = torch.log1p(torch.clamp(sigma * w_f, min=0.0) + eps)  # [F]
    N, F = X_cat.shape
    m = A_cat.shape[0]
    K_all = torch.empty((N, m), device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        Xb = X_cat[start:end]  # [B, F]
        logsum = torch.zeros((Xb.shape[0], m), device=device, dtype=torch.float32)
        for f in range(F):
            eq = (Xb[:, f].unsqueeze(1) == A_cat[:, f].unsqueeze(0)).to(torch.float32)  # [B, m]
            logsum += eq * log_f[f]
        # K = exp(logsum) - 1
        Kb = torch.expm1(logsum)
        K_all[start:end] = Kb
    return K_all


@torch.no_grad()
def field_weighted_match_kernel_XA(
    X_cat: torch.Tensor,   # [N, F]
    A_cat: torch.Tensor,   # [m, F]
    w_f: torch.Tensor,     # [F]
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Simple stabilizer kernel:
      K_match(x,a) = sum_f w_f * 1[x_f == a_f]
    Output: [N, m]
    """
    device = X_cat.device
    w_f = w_f.to(device=device, dtype=torch.float32)
    N, F = X_cat.shape
    m = A_cat.shape[0]
    K_all = torch.empty((N, m), device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        Xb = X_cat[start:end]
        s = torch.zeros((Xb.shape[0], m), device=device, dtype=torch.float32)
        for f in range(F):
            eq = (Xb[:, f].unsqueeze(1) == A_cat[:, f].unsqueeze(0)).to(torch.float32)
            s += eq * w_f[f]
        K_all[start:end] = s
    return K_all


@torch.no_grad()
def mC_kernel_from_t_XA(
    t_XA: torch.Tensor,   # [N, m] integer counts of matching fields
    c: int,
    F: int,
) -> torch.Tensor:
    """
    mC conjunctive kernel approximated by binomial coefficient:
      K = C(t, c)
    where t ∈ {0..F}.
    Vectorized via lookup table.
    """
    assert 0 <= c <= F
    device = t_XA.device

    # Precompute comb for all t in [0..F]
    comb = torch.zeros((F + 1,), device=device, dtype=torch.float32)
    for t in range(F + 1):
        if t >= c:
            comb[t] = float(math.comb(t, c))
    # Lookup
    t_clamped = torch.clamp(t_XA.to(torch.long), 0, F)
    return comb[t_clamped]


@torch.no_grad()
def diag_normalize_K_XA(K_XA: torch.Tensor, K_AA_diag: torch.Tensor, K_XX_diag: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Diagonal normalization:
      K_norm(x,a) = K(x,a) / sqrt(K(x,x) * K(a,a))
    For anchors: K_AA_diag [m]
    For samples: K_XX_diag [N] optional. If not provided, we approximate by using rowwise anchor self-similarity
    NOTE: For strictness, provide K(x,x); for many kernels we can compute it cheaply:
      - σ-mDNF: K(x,x) = prod_{f}(1+sigma*w_f) - 1  (since all fields match itself)
      - match: K(x,x)=sum w_f
      - mC: K(x,x)=C(F,c) for one-hot-per-field (all fields match itself)
    """
    eps = 1e-12
    denom_anchor = torch.sqrt(torch.clamp(K_AA_diag, min=eps)).unsqueeze(0)  # [1, m]
    if K_XX_diag is None:
        # weak fallback: normalize only by anchor diag
        return K_XA / denom_anchor
    denom_x = torch.sqrt(torch.clamp(K_XX_diag, min=eps)).unsqueeze(1)       # [N, 1]
    return K_XA / (denom_x * denom_anchor)


# -------------------------
# Anchor selection
# -------------------------

@torch.no_grad()
def random_anchors(X_cat: torch.Tensor, m: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    N = X_cat.shape[0]
    idx = torch.randperm(N, device=X_cat.device, generator=generator)[:m]
    return X_cat[idx].clone()


@torch.no_grad()
def farthest_point_anchors_hamming(
    X_cat: torch.Tensor,
    m: int,
    candidate_pool: int = 20000,
    seed: int = 0,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Approximate farthest-point sampling under Hamming distance on fields:
      dist(x,z) = F - #matches(fields)
    Uses a candidate pool to avoid O(N*m) scans each step.
    """
    device = X_cat.device
    N, F = X_cat.shape
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    pool = min(candidate_pool, N)
    cand_idx = torch.randperm(N, device=device, generator=g)[:pool]
    C = X_cat[cand_idx]  # [pool, F]

    # init anchor = random candidate
    a0 = C[torch.randint(low=0, high=pool, size=(1,), device=device, generator=g)]
    A = [a0.squeeze(0)]
    A_cat = torch.stack(A, dim=0)  # [1, F]

    # maintain min distance of candidates to selected anchors
    # compute matches count to anchors -> dist = F - matches
    t = match_count_matrix(C, A_cat, chunk_size=chunk_size).to(torch.int32)  # [pool, 1]
    min_dist = (F - t.squeeze(1)).to(torch.float32)  # [pool]

    for _ in range(1, m):
        # select candidate with max min_dist
        j = torch.argmax(min_dist)
        A.append(C[j].clone())
        A_cat = torch.stack(A, dim=0)  # [k, F]

        # update min_dist with distance to new anchor
        # compute matches between candidates and new anchor only
        new_anchor = A_cat[-1:].contiguous()
        t_new = match_count_matrix(C, new_anchor, chunk_size=chunk_size).to(torch.int32).squeeze(1)
        dist_new = (F - t_new).to(torch.float32)
        min_dist = torch.minimum(min_dist, dist_new)

    return torch.stack(A, dim=0)  # [m, F]


# -------------------------
# Nyström whitening
# -------------------------

@torch.no_grad()
def inv_sqrt_psd(M: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    Compute (M + ridge I)^(-1/2) for PSD matrix via eigendecomposition.
    M: [m, m]
    """
    m = M.shape[0]
    device = M.device
    I = torch.eye(m, device=device, dtype=M.dtype)
    Ms = (M + ridge * I).to(torch.float32)  # stabilize in fp32
    evals, evecs = torch.linalg.eigh(Ms)
    evals = torch.clamp(evals, min=1e-12)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    return inv_sqrt.to(M.dtype)


# -------------------------
# Streaming stats for leverage
# -------------------------

@torch.no_grad()
def accumulate_PhiT_Phi(
    encoder,
    N: int,
    d: int,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    encoder must yield (start, end, Phi_chunk [B, d]) via encoder.phi_chunks(...)
    Returns S = Phi^T Phi [d, d]
    """
    device = encoder.device
    S = torch.zeros((d, d), device=device, dtype=torch.float32)
    for _, _, Phi_b in encoder.phi_chunks(chunk_size=chunk_size):
        Pb = Phi_b.to(torch.float32)
        S += Pb.T @ Pb
    return S


@torch.no_grad()
def leverage_scores_streaming(
    encoder,
    ridge: float = 1e-2,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Compute ridge leverage scores:
      l_i = Phi_i (Phi^T Phi + ridge I)^(-1) Phi_i^T
    in 2 passes without storing Phi.
    """
    N = encoder.N
    d = encoder.d
    device = encoder.device

    S = accumulate_PhiT_Phi(encoder, N=N, d=d, chunk_size=chunk_size)
    I = torch.eye(d, device=device, dtype=torch.float32)
    M = torch.linalg.inv(S + ridge * I)  # [d, d]
    M = M.to(torch.float32)

    lev = torch.empty((N,), device=device, dtype=torch.float32)
    for start, end, Phi_b in encoder.phi_chunks(chunk_size=chunk_size):
        Pb = Phi_b.to(torch.float32)              # [B, d]
        tmp = Pb @ M                             # [B, d]
        lev[start:end] = (tmp * Pb).sum(dim=1)    # diag(P M P^T)
    # sanitize
    lev = torch.clamp(lev, min=0.0)
    return lev


@torch.no_grad()
def density_scores_streaming(
    encoder,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Simple representativeness proxy: sum of features (nonnegative).
    """
    N = encoder.N
    device = encoder.device
    dens = torch.empty((N,), device=device, dtype=torch.float32)
    for start, end, Phi_b in encoder.phi_chunks(chunk_size=chunk_size):
        dens[start:end] = Phi_b.to(torch.float32).sum(dim=1)
    return torch.clamp(dens, min=0.0)


@torch.no_grad()
def energy_scores_streaming(
    encoder,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Row energy: ||Phi_i||^2.
    """
    N = encoder.N
    device = encoder.device
    en = torch.empty((N,), device=device, dtype=torch.float32)
    for start, end, Phi_b in encoder.phi_chunks(chunk_size=chunk_size):
        Pb = Phi_b.to(torch.float32)
        en[start:end] = (Pb * Pb).sum(dim=1)
    return torch.clamp(en, min=0.0)


def _normalize01(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.to(torch.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


@torch.no_grad()
def sample_top_or_prob(
    scores: torch.Tensor,
    k: int,
    mode: str = "prob",  # "prob" or "topk"
    temperature: float = 1.0,
    clip_top_frac: float = 0.001,
    seed: int = 0,
) -> torch.Tensor:
    """
    Returns indices [k] sampled without replacement.
    For probabilistic sampling: p_i ∝ (scores_i)^(1/temperature).
    Clipping helps avoid dominance by outliers.
    """
    device = scores.device
    N = scores.numel()
    scores = scores.to(torch.float32)

    # clip extreme top to stabilize (winsorize)
    if clip_top_frac is not None and clip_top_frac > 0:
        q = max(1, int((1.0 - clip_top_frac) * N))
        thr = torch.kthvalue(scores, k=q).values
        scores = torch.minimum(scores, thr)

    if mode == "topk":
        return torch.topk(scores, k=min(k, N), largest=True).indices

    # prob mode
    s = torch.clamp(scores, min=1e-12)
    if temperature != 1.0:
        s = s.pow(1.0 / max(temperature, 1e-6))
    p = s / s.sum()

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    idx = torch.multinomial(p, num_samples=min(k, N), replacement=False, generator=g)
    return idx


# -------------------------
# Kernel encoder (multi-kernel, Nyström)
# -------------------------

@dataclass
class KernelSpec:
    name: str
    kind: str  # "mdnf" | "match" | "mC"
    sigma: float = 1.0
    c: int = 2
    weight: float = 1.0  # for mixing; we'll normalize anyway


class MultiKernelEncoder:
    """
    Builds Phi(x) = concat_k  K_k(x, A) * inv_sqrt(K_k(A,A)+ridge I)  (Nyström whitening)
    """
    def __init__(
        self,
        X_cat: torch.Tensor,          # [N, F]
        A_cat: torch.Tensor,          # [m, F]
        specs: List[KernelSpec],
        w_f: torch.Tensor,            # [F]
        nystrom_ridge: float = 1e-3,
        chunk_size: int = 65536,
    ):
        self.X_cat = X_cat
        self.A_cat = A_cat
        self.specs = specs
        self.w_f = w_f
        self.nystrom_ridge = nystrom_ridge
        self.chunk_size = chunk_size

        self.device = X_cat.device
        self.N, self.F = X_cat.shape
        self.m = A_cat.shape[0]

        # Precompute anchor-anchor blocks and their invsqrt for each kernel
        self._KAA_invsqrt: Dict[str, torch.Tensor] = {}
        self._KAA_diag: Dict[str, torch.Tensor] = {}
        self._x_self_diag: Dict[str, torch.Tensor] = {}  # K(x,x) per kernel (cheap)
        self._prepare()

        self.d = self.m * len(self.specs)

    @torch.no_grad()
    def _prepare(self):
        # Precompute t_AA once (cheap since m small)
        t_AA = match_count_matrix(self.A_cat, self.A_cat, chunk_size=min(self.chunk_size, 8192)).to(torch.int32)  # [m, m]

        # sample self diag for X for kernels where cheap closed form exists
        sum_w = self.w_f.to(self.device, dtype=torch.float32).sum()

        for spec in self.specs:
            if spec.kind == "mdnf":
                # K(x,x) = prod_f (1 + sigma*w_f) - 1   since all fields match itself
                log_f = torch.log1p(torch.clamp(spec.sigma * self.w_f.to(self.device, dtype=torch.float32), min=0.0) + 1e-12)
                K_xx = torch.expm1(log_f.sum()).repeat(self.N)  # scalar repeated
                # anchors K_AA
                K_AA = weighted_logprod_mdnf_kernel_XA(self.A_cat, self.A_cat, self.w_f, sigma=spec.sigma, chunk_size=min(self.chunk_size, 8192))
            elif spec.kind == "match":
                # K(x,x)=sum w_f
                K_xx = sum_w.repeat(self.N)
                K_AA = field_weighted_match_kernel_XA(self.A_cat, self.A_cat, self.w_f, chunk_size=min(self.chunk_size, 8192))
            elif spec.kind == "mC":
                # For field-wise representation, each sample matches itself on all F fields => t=F
                # K(x,x)=C(F,c)
                K_xx = torch.tensor(float(math.comb(self.F, spec.c)) if self.F >= spec.c else 0.0, device=self.device).repeat(self.N)
                K_AA = mC_kernel_from_t_XA(t_AA, c=spec.c, F=self.F)
            else:
                raise ValueError(f"Unknown kernel kind: {spec.kind}")

            # Diagonal normalization (important for mixing kernels)
            K_AA_diag = torch.diag(K_AA).to(torch.float32)
            self._KAA_diag[spec.name] = K_AA_diag
            self._x_self_diag[spec.name] = K_xx.to(torch.float32)

            # Normalize K_AA itself to have diag ~1 before invsqrt to stabilize
            K_AA_norm = diag_normalize_K_XA(K_AA, K_AA_diag, K_XX_diag=K_AA_diag)
            # Optional trace normalization to align scales further
            tr = torch.trace(K_AA_norm.to(torch.float32)).clamp(min=1e-12)
            K_AA_norm = (K_AA_norm.to(torch.float32) / tr).to(torch.float32)

            invsqrt = inv_sqrt_psd(K_AA_norm, ridge=self.nystrom_ridge).to(torch.float32)
            self._KAA_invsqrt[spec.name] = invsqrt

    @torch.no_grad()
    def _K_XA(self, spec: KernelSpec, Xb: torch.Tensor) -> torch.Tensor:
        if spec.kind == "mdnf":
            K = weighted_logprod_mdnf_kernel_XA(Xb, self.A_cat, self.w_f, sigma=spec.sigma, chunk_size=Xb.shape[0])
        elif spec.kind == "match":
            K = field_weighted_match_kernel_XA(Xb, self.A_cat, self.w_f, chunk_size=Xb.shape[0])
        elif spec.kind == "mC":
            # Use t_XA -> binomial
            t = match_count_matrix(Xb, self.A_cat, chunk_size=Xb.shape[0]).to(torch.int32)  # [B, m]
            K = mC_kernel_from_t_XA(t, c=spec.c, F=self.F)
        else:
            raise ValueError(spec.kind)

        # Diagonal normalization against sample self and anchor self
        K = diag_normalize_K_XA(K.to(torch.float32), self._KAA_diag[spec.name], self._x_self_diag[spec.name][0:Xb.shape[0]].new_full((Xb.shape[0],), self._x_self_diag[spec.name][0].item()))
        # Trace normalization factor (match training-time KAA normalization): use same tr as in _prepare
        # For simplicity: we already normalized KAA by its trace; here we do the same by anchor count.
        # (This is approximate but works in practice; if you want exact, store tr per kernel.)
        K = K / max(float(self.m), 1.0)
        return K

    @torch.no_grad()
    def phi_chunks(self, X_cat_custom: Optional[torch.Tensor] = None, chunk_size: Optional[int] = None):
        if X_cat_custom is None:
            X_cat_custom = self.X_cat
        N = X_cat_custom.shape[0]
        cs = chunk_size or self.chunk_size
        for start in range(0, N, cs):
            end = min(N, start + cs)
            Xb = X_cat_custom[start:end]
            feats = []
            for spec in self.specs:
                Kb = self._K_XA(spec, Xb)  # [B, m]
                # Nyström whitening
                invsqrt = self._KAA_invsqrt[spec.name]  # [m, m]
                Phi_b = (Kb.to(torch.float32) @ invsqrt) * spec.weight
                feats.append(Phi_b)
            Phi = torch.cat(feats, dim=1)  # [B, m*q]
            yield start, end, Phi

    @torch.no_grad()
    def get_train_embeddings(self, idx_cov, device, chunk_size: Optional[int] = None):
        d = self.d
        Phi_cov = torch.empty((idx_cov.numel(), d), device=device, dtype=torch.float32)
        # mapping from global index -> position in Phi_cov
        pos = -torch.ones((self.N,), device=device, dtype=torch.int64)
        pos[idx_cov] = torch.arange(idx_cov.numel(), device=device, dtype=torch.int64)

        for start, end, Phi_b in self.phi_chunks(chunk_size=chunk_size):
            idx_range = torch.arange(start, end, device=device)
            hit = pos[idx_range]
            sel = hit >= 0
            if sel.any():
                Phi_cov[hit[sel]] = Phi_b[sel].to(torch.float32)

        return Phi_cov

    @torch.no_grad()
    def get_custom_embeddings(self, X_cat_custom: torch.Tensor, device, chunk_size: Optional[int] = None):
        Phi = torch.empty((X_cat_custom.shape[0], self.d), device=device, dtype=torch.float32)

        for start, end, Phi_b in self.phi_chunks(X_cat_custom=X_cat_custom, chunk_size=chunk_size):
            Phi[start:end] = Phi_b.to(torch.float32)

        return Phi


@dataclass
class HardModeConfig:
    coverage_frac: float = 0.6
    clf_ridge: float = 1e-2
    hard_mode: Literal["hinge", "margin"] = "hinge"
    hard_clip_top_frac: float = 0.002



class KernelSampler:
    """
    Семплирование на основе ядерных функций
    """

    def __init__(
        self,
        sample_size: int,
        kernels: List[dict],
        energy_score: float,
        density_score: float,
        leverage_score: float,
        random_state: int = 42,
        anchors: int = 512,
        anchor_mode: Literal['farthest', 'random'] = "farthest",
        w_f: Optional[torch.Tensor] = None,
        chunk_size: int = 65536,
        leverage_ridge: float = 1e-2,
        use_hard_mode: bool = False,
        hard_mode_config: Optional[HardModeConfig] = None
    ):
        self.sample_size = sample_size
        self.kernels = kernels
        self.energy_score = energy_score
        self.density_score = density_score
        self.leverage_score = leverage_score
        self.random_state = random_state
        self.anchors = anchors
        self.anchor_mode = anchor_mode
        self.w_f = w_f
        self.chunk_size = chunk_size
        self.leverage_ridge = leverage_ridge
        self.use_hard_mode = use_hard_mode
        self.hard_mode_config = hard_mode_config

        self.encoder = None

    @staticmethod
    @torch.no_grad()
    def _ridge_classifier_fit(
        Phi_S: torch.Tensor,  # [S, d]
        y_S: torch.Tensor,    # [S] in {-1, +1}
        ridge: float = 1e-2,
    ) -> torch.Tensor:
        """
        Closed-form ridge regression classifier:
          w = (Phi^T Phi + ridge I)^(-1) Phi^T y
        Returns w [d]
        """
        Phi = Phi_S.to(torch.float32)
        y = y_S.to(torch.float32).view(-1, 1)
        d = Phi.shape[1]
        A = Phi.T @ Phi + ridge * torch.eye(d, device=Phi.device, dtype=torch.float32)
        b = Phi.T @ y
        w = torch.linalg.solve(A, b).view(-1)
        return w

    @torch.no_grad()
    def get_embeddings(
        self,
        X_cat: torch.Tensor,
        device,
        chunk_size=65536,
    ) -> torch.Tensor:
        return self.encoder.get_custom_embeddings(X_cat_custom=X_cat, device=device, chunk_size=chunk_size)

    def fit_sample(self, X_cat: torch.Tensor, y: Optional[torch.Tensor] = None):
        device = X_cat.device
        N, F = X_cat.shape
        if self.w_f is None:
            self.w_f = torch.ones((F,), device=device, dtype=torch.float32)

        if self.anchor_mode == "random":
            g = torch.Generator(device=device)
            g.manual_seed(self.random_state)
            A_cat = random_anchors(X_cat, m=self.anchors, generator=g)
        elif self.anchor_mode == "farthest":
            A_cat = farthest_point_anchors_hamming(
                X_cat, m=self.anchors, candidate_pool=min(20000, N), seed=self.random_state,
                chunk_size=min(self.chunk_size, 32768)
            )
        else:
            raise ValueError("anchor_mode must be random or farthest")

        specs: List[KernelSpec] = [KernelSpec(**kernel) for kernel in self.kernels]
        self.encoder = MultiKernelEncoder(X_cat=X_cat, A_cat=A_cat, specs=specs, w_f=self.w_f,
                                          nystrom_ridge=1e-3, chunk_size=self.chunk_size)

        score_components = [
            (self.energy_score, energy_scores_streaming, (self.encoder, self.chunk_size)),
            (self.density_score, density_scores_streaming, (self.encoder, self.chunk_size)),
            (self.leverage_score, leverage_scores_streaming, (self.encoder, self.leverage_ridge, self.chunk_size)),
        ]
        score = None

        for weight, func, args in score_components:
            if weight is None:
                continue

            component_score = weight * _normalize01(func(*args))

            if score is None:
                score = torch.zeros_like(component_score)

            score += component_score

        if score is None:
            raise ValueError("At least one score weight must be specified")

        if not self.use_hard_mode:
            idx = sample_top_or_prob(score, k=self.sample_size, mode="prob", temperature=1.0, seed=self.random_state)
            return idx, A_cat, self.encoder.get_train_embeddings(idx, device, chunk_size=self.chunk_size)

        if self.use_hard_mode:
            if self.hard_mode_config is None or y is None:
                raise ValueError("hard_mode_config and y must be specified in hard mode")
            sample_size_cov = int(round(self.sample_size * self.hard_mode_config.coverage_frac))
            idx_cov = sample_top_or_prob(score, k=sample_size_cov, mode="prob", temperature=1.0, seed=self.random_state)

            # Stage 2: hard/uncertainty refinement
            sample_size_hard = self.sample_size - sample_size_cov
            if sample_size_hard <= 0:
                return idx_cov, A_cat

            # Train proxy ridge classifier on coverage set (works only if y reasonably clean)
            y = y.to(device=device, dtype=torch.float32).view(-1)
            y_cov = y[idx_cov]

            # Build Phi for idx_cov only (one chunk pass, but simplest: compute all and index would be expensive)
            # We'll compute Phi for all in streaming and cache only cov rows.
            Phi_cov = self.encoder.get_train_embeddings(idx_cov, device=device, chunk_size=self.chunk_size)
            w = self._ridge_classifier_fit(Phi_cov, y_cov, ridge=self.hard_mode_config.clf_ridge)  # [d]

            # Compute hinge loss / uncertainty for all points (streaming)
            hard_score = torch.empty((N,), device=device, dtype=torch.float32)
            for start, end, Phi_b in self.encoder.phi_chunks(chunk_size=self.chunk_size):
                Pb = Phi_b.to(torch.float32)
                f = Pb @ w  # [B]
                yb = y[start:end]
                if self.hard_mode_config.hard_mode == "hinge":
                    loss = torch.relu(1.0 - yb * f)
                    hard_score[start:end] = loss
                elif self.hard_mode_config.hard_mode == "margin":
                    hard_score[start:end] = -torch.abs(f)  # smaller abs => harder (note negative)
                else:
                    raise ValueError(self.hard_mode_config.hard_mode)

            # Convert margin to positive score if needed
            if self.hard_mode_config.hard_mode == "margin":
                hard_score = 1.0 - _normalize01(torch.abs(hard_score))

            # Avoid duplicates with coverage set
            mask = torch.ones((N,), device=device, dtype=torch.bool)
            mask[idx_cov] = False
            hard_score = torch.clamp(hard_score, min=0.0) * mask.to(torch.float32)

            idx_hard = sample_top_or_prob(hard_score, k=sample_size_hard, mode="prob", temperature=1.0, seed=self.random_state + 2,
                                          clip_top_frac=self.hard_mode_config.hard_clip_top_frac)

            idx = torch.unique(torch.cat([idx_cov, idx_hard], dim=0))[:self.sample_size]
            return idx, A_cat, self.encoder.get_train_embeddings(idx, device, chunk_size=self.chunk_size)
