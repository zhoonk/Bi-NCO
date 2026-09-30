"""Experiment options for the v4 ablation study.

Every variant is expressed as a set of overrides on top of the M0 (Bi-NCO)
defaults, so that all variants share the same code path and differ only in the
options listed here.
"""

DEFAULTS = {
    'direction_mode': 'bi',        # 'bi' | 'fwd' | 'bwd'
    'decoder_wiring': 'exchange',  # 'exchange' | 'shared'  (the original PFSP code already uses 'exchange')
    'encoder_coupling': 'cross',   # 'cross': each stream attends to the other | 'self': to itself (M9)
    'architecture': 'desd',        # 'desd': dual encoder streams (Bi-NCO) | 'sedd': single encoder, dual decoder projections
    'pseudo_label': 'separate',    # 'separate' | 'global'
    'weighting': 'adaptive',       # 'adaptive' | 'uniform' | 'clipped'
    'clip_value': None,            # upper bound of alpha for 'clipped' (M8); must be given (--clip_value)
    'loss_type': 'si',             # 'si' (self-improvement) | 'pg' (REINFORCE, shared baseline)
    'transpose_aug': False,        # ATSP only: randomly transpose training cost matrices
}

VARIANTS = {
    'M0': {},                                                 # Bi-NCO
    'M1': {'direction_mode': 'fwd'},                          # forward-only training
    'M2': {'direction_mode': 'bwd'},                          # backward-only training
    'M3': {'decoder_wiring': 'shared'},                       # direction token, no role exchange
    # M4 (transposed-cost augmentation) is defined for the ATSP only
    'M5': {'pseudo_label': 'global'},                         # global best-of-2N pseudo-label
    'M6': {'weighting': 'uniform'},                           # uniform pseudo-label weights
    'M7': {'loss_type': 'pg'},                                # policy gradient instead of self-improvement
    'M8': {'weighting': 'clipped'},                           # standardized weight clipped at clip_value
    'M9': {'encoder_coupling': 'self'},                       # uncoupled encoder streams (PFSP only)
    'SEDD': {'architecture': 'sedd'},                         # shared encoder + role-specific decoder heads (PFSP only)
}


def resolve_variant(name):
    if name not in VARIANTS:
        raise ValueError('unknown variant {}; choose from {}'.format(name, sorted(VARIANTS)))
    options = dict(DEFAULTS)
    options.update(VARIANTS[name])
    return options


def direction_split(params):
    """Return (n_fwd, n_bwd), the number of rollouts per instance in each direction.

    The total is 2 * trajectory_size for every direction_mode, so that all variants
    use the same number of trajectories per instance. Explicit 'n_fwd' and 'n_bwd'
    entries (used at test time, e.g. for greedy decoding) take precedence.
    """
    if 'n_fwd' in params and 'n_bwd' in params:
        return int(params['n_fwd']), int(params['n_bwd'])
    n = int(params['trajectory_size'])
    mode = params.get('direction_mode', 'bi')
    if mode == 'bi':
        return n, n
    if mode == 'fwd':
        return 2 * n, 0
    if mode == 'bwd':
        return 0, 2 * n
    raise ValueError('unknown direction_mode {}'.format(mode))
