import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sensor_groups(acc_axes, gyro_axes, mag_axes, acc_mod, gyro_mod, mag_mod):
    """Define sensor groups based on hyperparameters."""
    return {
        "axes": [acc_axes, gyro_axes, mag_axes],
        "modality": [acc_mod + gyro_mod, mag_mod]
    }

def group_ortho_loss(group_feats: list) -> torch.Tensor:
    """Compute orthogonal loss between group features."""
    G = [F.normalize(g.flatten(1), dim=1) for g in group_feats]
    loss = 0.0
    n = len(G)
    for i in range(n):
        for j in range(i + 1, n):
            cov = G[i].t() @ G[j]
            loss += cov.pow(2).mean()
    return loss / max(1, (n * (n - 1)) / 2)

def fft_filter(x, cutoff_hz, fs, btype='low'):
    """Apply FFT-based frequency filter."""
    B, C, T = x.shape
    freqs = torch.fft.fftfreq(T, d=1/fs).to(x.device)
    x_fft = torch.fft.fft(x, dim=-1)
    if btype == 'low':
        mask = torch.abs(freqs) <= cutoff_hz
    else:
        mask = torch.abs(freqs) > cutoff_hz
    mask = mask.view(1, 1, -1).expand(B, C, -1)
    x_fft_filtered = x_fft * mask
    return torch.fft.ifft(x_fft_filtered, dim=-1).real

def compute_gravity_gyro_consistency(total_acc, gyro, gravity_est, fs, eps=1e-6):
    """Compute consistency loss between gravity and gyro."""
    dt = 1.0 / fs
    B, C, T = total_acc.shape
    num_sensors = min(total_acc.shape[1] // 3, gyro.shape[1] // 3)
    total_sensor_loss = 0.0
    ux = torch.tensor([1., 0., 0.], device=total_acc.device)
    uy = torch.tensor([0., 1., 0.], device=total_acc.device)
    uz = torch.tensor([0., 0., 1.], device=total_acc.device)
    for i in range(num_sensors):
        start_idx, end_idx = i * 3, (i + 1) * 3
        sensor_gravity_est = gravity_est[:, start_idx:end_idx, :]
        sensor_gyro = gyro[:, start_idx:end_idx, :]
        gravity_norm = F.normalize(sensor_gravity_est, dim=1, eps=eps)
        loss_per_sensor = 0.0
        for t in range(1, T):
            gravity_prev = gravity_norm[:, :, t - 1]
            gravity_curr = gravity_norm[:, :, t]
            gyro_angular_vel = sensor_gyro[:, :, t - 1] * dt
            gravity_predicted = gravity_prev.clone()
            axis_x = torch.cross(gravity_prev, ux.expand_as(gravity_prev), dim=1)
            axis_y = torch.cross(gravity_prev, uy.expand_as(gravity_prev), dim=1)
            axis_z = torch.cross(gravity_prev, uz.expand_as(gravity_prev), dim=1)
            rotation_x = gyro_angular_vel[:, 0:1]
            rotation_y = gyro_angular_vel[:, 1:2]
            rotation_z = gyro_angular_vel[:, 2:3]
            gravity_predicted = gravity_predicted + rotation_x * axis_x + rotation_y * axis_y + rotation_z * axis_z
            gravity_predicted = F.normalize(gravity_predicted, dim=1, eps=eps)
            consistency_loss = F.mse_loss(gravity_predicted, gravity_curr)
            loss_per_sensor += consistency_loss
        total_sensor_loss += loss_per_sensor / (T - 1)
    return total_sensor_loss / num_sensors

def physics_guided_loss(x, gravity_scale, fs, hp_cut, lp_cut, eps, acc_indices, gyro_indices):
    """Physics-guided loss for sensor data."""
    total_acc = x[:, acc_indices, :]
    gyro = x[:, gyro_indices, :]
    low_freq = fft_filter(total_acc, cutoff_hz=lp_cut, fs=fs, btype='low')
    L_grav = compute_gravity_gyro_consistency(total_acc, gyro, low_freq, fs)
    acc_high = fft_filter(total_acc, cutoff_hz=hp_cut, fs=fs, btype='high')
    gyro_high = fft_filter(gyro, cutoff_hz=hp_cut, fs=fs, btype='high')
    acc_activity = (acc_high ** 2).sum(dim=1)
    gyro_activity = (gyro_high ** 2).sum(dim=1)
    acc_norm = (acc_activity - acc_activity.mean(dim=-1, keepdims=True)) / (acc_activity.std(dim=-1, keepdims=True) + eps)
    gyro_norm = (gyro_activity - gyro_activity.mean(dim=-1, keepdims=True)) / (gyro_activity.std(dim=-1, keepdims=True) + eps)
    L_ag = F.mse_loss(acc_norm, gyro_norm)
    acc_temporal = acc_high.flatten(1)
    gyro_temporal = gyro_high.flatten(1)
    acc_t_norm = F.normalize(acc_temporal, dim=1)
    gyro_t_norm = F.normalize(gyro_temporal, dim=1)
    correlation = (acc_t_norm * gyro_t_norm).sum(dim=1).mean()
    L_ag_corr = 1.0 - correlation
    L_ag_combined = L_ag + 2.0 * L_ag_corr
    acc_jerk = torch.diff(total_acc, dim=-1)
    gyro_jerk = torch.diff(gyro, dim=-1)
    L_jerk = (acc_jerk ** 2).mean() + (gyro_jerk ** 2).mean()
    return L_grav, L_ag_combined, L_jerk

class ELKBlock(nn.Module):
    """Expanded Large Kernel Block for feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        padding_large1 = kernel_size // 2
        kernel_size_large2 = kernel_size - 2
        padding_large2 = (kernel_size - 2) // 2
        if deploy:
            self.reparam_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding_large1, groups=in_channels, bias=True)
        else:
            self.dw_large1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding_large1, groups=in_channels, bias=False)
            self.bn_large1 = nn.BatchNorm1d(in_channels)
            self.dw_large2 = nn.Conv1d(in_channels, in_channels, kernel_size_large2, padding=padding_large2, groups=in_channels, bias=False)
            self.bn_large2 = nn.BatchNorm1d(in_channels)
            self.dw_small1 = nn.Conv1d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False)
            self.bn_small1 = nn.BatchNorm1d(in_channels)
            self.dw_small2 = nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            self.bn_small2 = nn.BatchNorm1d(in_channels)
            self.bn_id = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        if not self.deploy:
            x = self.bn_large1(self.dw_large1(x)) + self.bn_large2(self.dw_large2(x)) + \
                self.bn_small1(self.dw_small1(x)) + self.bn_small2(self.dw_small2(x)) + self.bn_id(x)
        else:
            x = self.reparam_conv(x)
        return self.pointwise(self.activation(x))

    def reparameterize(self):
        if self.deploy:
            return
        def _fuse(conv, bn):
            if conv is None:
                kernel = torch.zeros((self.in_channels, 1, self.kernel_size), dtype=bn.weight.dtype, device=bn.weight.device)
                kernel[:, 0, self.kernel_size // 2] = 1.0
                conv_bias = torch.zeros_like(bn.running_mean)
            else:
                kernel = conv.weight
                conv_bias = torch.zeros_like(bn.running_mean)
            std = (bn.running_var + bn.eps).sqrt()
            return kernel * (bn.weight / std).reshape(-1, 1, 1), (bn.weight / std) * (conv_bias - bn.running_mean) + bn.bias

        w_l1, b_l1 = _fuse(self.dw_large1, self.bn_large1)
        w_l2, b_l2 = _fuse(self.dw_large2, self.bn_large2)
        w_s1, b_s1 = _fuse(self.dw_small1, self.bn_small1)
        w_s2, b_s2 = _fuse(self.dw_small2, self.bn_small2)
        w_id, b_id = _fuse(None, self.bn_id)

        w_l2 = F.pad(w_l2, ((self.kernel_size - (self.kernel_size - 2)) // 2,) * 2)
        w_s1 = F.pad(w_s1, ((self.kernel_size - 5) // 2,) * 2)
        w_s2 = F.pad(w_s2, ((self.kernel_size - 3) // 2,) * 2)

        self.reparam_conv = nn.Conv1d(self.in_channels, self.in_channels, self.kernel_size, padding=self.kernel_size // 2, groups=self.in_channels, bias=True).to(w_l1.device)
        self.reparam_conv.weight.data = w_l1 + w_l2 + w_s1 + w_s2 + w_id
        self.reparam_conv.bias.data = b_l1 + b_l2 + b_s1 + b_s2 + b_id
        self.deploy = True
        for attr in ['dw_large1', 'bn_large1', 'dw_large2', 'bn_large2', 'dw_small1', 'bn_small1', 'dw_small2', 'bn_small2', 'bn_id']:
            if hasattr(self, attr):
                delattr(self, attr)

class StructureAwareEmbeddingELK(nn.Module):
    """Structure-aware embedding using ELK blocks."""
    def __init__(self, sensor_groups: list, d_model: int, elk_kernel_size: int):
        super().__init__()
        self.sensor_groups = sensor_groups
        self.group_embeds = nn.ModuleList([
            nn.Sequential(
                ELKBlock(len(group), d_model // 2, kernel_size=elk_kernel_size),
                nn.GELU()
            ) for group in sensor_groups
        ])
        self.mixer = nn.Sequential(
            nn.Conv1d((d_model // 2) * len(sensor_groups), d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        group_embeddings = [embed(x[:, group, :]) for embed, group in zip(self.group_embeds, self.sensor_groups)]
        return self.mixer(torch.cat(group_embeddings, dim=1)), group_embeddings

class TimeAwareCrossAttention(nn.Module):
    """Time-aware cross-attention module."""
    def __init__(self, dim, n_heads, dropout, max_rel, time_sigma):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.max_rel = max_rel
        self.time_sigma = time_sigma
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_rel + 1))
        nn.init.normal_(self.rel_bias, std=0.01)
        self.mask_scale = nn.Parameter(torch.tensor(0.5))

    def _relative_time_bias(self, T, Lq, device, tau_q=None):
        t_k = torch.arange(T, device=device).view(1, 1, T)
        if tau_q is None:
            tau_q = torch.linspace(0, T - 1, steps=Lq, device=device)
        tau_q = tau_q.view(1, Lq, 1)
        dt = t_k - tau_q
        dt_clip = dt.clamp(-self.max_rel, self.max_rel).long() + self.max_rel
        bias = self.rel_bias[dt_clip]
        gauss = torch.exp(-0.5 * (dt / self.time_sigma) ** 2)
        return bias + torch.log(gauss + 1e-6)

    def forward(self, q, kv, tau_q=None):
        time_bias = self._relative_time_bias(kv.shape[1], q.shape[1], q.device, tau_q)
        attn_mask = (self.mask_scale * time_bias).squeeze(0)
        out, _ = self.mha(q, kv, kv, attn_mask=attn_mask)
        return out

class MLP(nn.Module):
    """Multi-layer perceptron module."""
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class PerceiverBackbone(nn.Module):
    """Perceiver backbone for latent processing."""
    def __init__(self, d_in, d_latent, n_latents, n_blocks, n_heads, mlp_ratio, dropout, time_sigma, max_rel):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_latent)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_latent) * 0.02)
        self.register_buffer("latent_anchors", None, persistent=False)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_x': nn.LayerNorm(d_latent),
                'xattn': TimeAwareCrossAttention(d_latent, n_heads, dropout, max_rel, time_sigma),
                'mlp_x': MLP(d_latent, int(d_latent * mlp_ratio), dropout),
                'norm_s': nn.LayerNorm(d_latent),
                'self_attn': SelfAttention(d_latent, n_heads, dropout),
                'mlp_s': MLP(d_latent, int(d_latent * mlp_ratio), dropout),
            }) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_latent)
        self.out_dim = d_latent

    def forward(self, tokens_bt_d):
        B, T, _ = tokens_bt_d.shape
        tokens = self.proj_in(tokens_bt_d)
        latents = self.latents.expand(B, -1, -1)
        if self.latent_anchors is None or self.latent_anchors.numel() != latents.shape[1]:
            self.latent_anchors = torch.linspace(0, T - 1, steps=latents.shape[1], device=latents.device)
        for blk in self.blocks:
            latents = latents + blk['xattn'](blk['norm_x'](latents), tokens, tau_q=self.latent_anchors)
            latents = latents + blk['mlp_x'](blk['norm_x'](latents))
            latents = latents + blk['self_attn'](blk['norm_s'](latents))
            latents = latents + blk['mlp_s'](blk['norm_s'](latents))
        return self.norm_out(latents).mean(dim=1)

class P2Perceiver(nn.Module):
    """Main model: P2-Perceiver"""
    def __init__(self, n_channels, seq_length, d_model, n_classes, group_type, use_structure_embed, use_perceiver, elk_kernel_size,
                 d_latent, n_latents, n_blocks, n_heads, mlp_ratio, dropout, time_sigma, max_rel, acc_axes, gyro_axes, mag_axes, acc_mod, gyro_mod, mag_mod):
        super().__init__()
        self.use_structure_embed = use_structure_embed
        self.use_perceiver = use_perceiver
        self.gravity_scale = nn.Parameter(torch.tensor(1.0))
        sensor_groups = get_sensor_groups(acc_axes, gyro_axes, mag_axes, acc_mod, gyro_mod, mag_mod)[group_type]
        if use_structure_embed:
            self.embed = StructureAwareEmbeddingELK(sensor_groups, d_model, elk_kernel_size)
        else:
            self.embed = nn.Sequential(
                ELKBlock(n_channels, d_model, elk_kernel_size),
                nn.GELU()
            )
        self.pos_encoding = nn.Parameter(torch.zeros(1, d_model, seq_length))
        if use_perceiver:
            self.backbone = PerceiverBackbone(d_in=d_model, d_latent=d_latent, n_latents=n_latents, n_blocks=n_blocks,
                                              n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout, time_sigma=time_sigma, max_rel=max_rel)
            self.norm = nn.LayerNorm(self.backbone.out_dim)
            feat_dim = self.backbone.out_dim
        else:
            self.backbone = None
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.norm = nn.LayerNorm(d_model)
            feat_dim = d_model
        self.classifier = nn.Linear(feat_dim, n_classes)

    def extract_features(self, x):
        if self.use_structure_embed:
            h, inter = self.embed(x)
        else:
            h = self.embed(x)
            inter = None
        if h.shape[-1] == self.pos_encoding.shape[-1]:
            h = h + self.pos_encoding
        if self.use_perceiver:
            pooled = self.norm(self.backbone(h.transpose(1, 2)))
        else:
            pooled = self.pool(h).squeeze(-1)
            pooled = self.norm(pooled)
        return (pooled, inter) if self.training and self.use_structure_embed else pooled

    def forward(self, x):
        output = self.extract_features(x)
        if self.training and self.use_structure_embed:
            features, inter = output
            logits = self.classifier(features)
            return logits, inter
        else:
            features = output
            logits = self.classifier(features)
            return logits

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, ELKBlock):
                m.reparameterize()
              
def parse_args():
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    # General model params
    parser.add_argument('--n_channels', type=int, default=21, help='Number of input channels')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--group_type', type=str, default='axes', help='Sensor group type (axes or modality)')
    parser.add_argument('--use_structure_embed', action='store_true', help='Use structure-aware embedding')
    parser.add_argument('--use_perceiver', action='store_true', help='Use Perceiver backbone')
    parser.add_argument('--elk_kernel_size', type=int, default=31, help='ELK kernel size')
    parser.add_argument('--d_latent', type=int, default=128, help='Latent dimension')
    parser.add_argument('--n_latents', type=int, default=32, help='Number of latents')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=1, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--time_sigma', type=float, default=32.0, help='Time sigma for attention')
    parser.add_argument('--max_rel', type=int, default=50, help='Max relative position')
    
    # Sensor feature indices
    parser.add_argument('--acc_axes', type=lambda s: [int(i) for i in s.split(',')], default=[0,3,6,9,12,15,18], help='Accelerometer axes indices')
    parser.add_argument('--gyro_axes', type=lambda s: [int(i) for i in s.split(',')], default=[1,4,7,10,13,16,19], help='Gyroscope axes indices')
    parser.add_argument('--mag_axes', type=lambda s: [int(i) for i in s.split(',')], default=[2,5,8,11,14,17,20], help='Magnetometer axes indices')
    parser.add_argument('--acc_mod', type=lambda s: [int(i) for i in s.split(',')], default=[0,1,2,6,7,8,12,13,14,15,16,17], help='Accelerometer modality indices')
    parser.add_argument('--gyro_mod', type=lambda s: [int(i) for i in s.split(',')], default=[3,4,5], help='Gyroscope modality indices')  # Adjusted default if needed
    parser.add_argument('--mag_mod', type=lambda s: [int(i) for i in s.split(',')], default=[9,10,11,18,19,20], help='Magnetometer modality indices')
    
    # Physics loss params
    parser.add_argument('--fs', type=int, default=50, help='Sampling frequency')
    parser.add_argument('--hp_cut', type=float, default=1.5, help='High-pass cutoff')
    parser.add_argument('--lp_cut', type=float, default=0.5, help='Low-pass cutoff')
    parser.add_argument('--eps', type=float, default=1e-6, help='Epsilon for normalization')
    parser.add_argument('--acc_indices', type=lambda s: [int(i) for i in s.split(',')], default=[6,7,8,15,16,17], help='Accelerometer indices for physics loss')
    parser.add_argument('--gyro_indices', type=lambda s: [int(i) for i in s.split(',')], default=[9,10,11,18,19,20], help='Gyroscope indices for physics loss')
    
    # Transformer specific
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Example instantiation of the main model
    model = P2Perceiver(
        n_channels=args.n_channels,
        seq_length=args.seq_length,
        d_model=args.d_model,
        n_classes=args.n_classes,
        group_type=args.group_type,
        use_structure_embed=args.use_structure_embed,
        use_perceiver=args.use_perceiver,
        elk_kernel_size=args.elk_kernel_size,
        d_latent=args.d_latent,
        n_latents=args.n_latents,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        time_sigma=args.time_sigma,
        max_rel=args.max_rel,
        acc_axes=args.acc_axes,
        gyro_axes=args.gyro_axes,
        mag_axes=args.mag_axes,
        acc_mod=args.acc_mod,
        gyro_mod=args.gyro_mod,
        mag_mod=args.mag_mod
    )
    print(model)
    # Example physics loss call (dummy input)
    dummy_x = torch.randn(1, args.n_channels, args.seq_length)
    gravity_scale = torch.tensor(1.0)
    L_grav, L_ag_combined, L_jerk = physics_guided_loss(
        dummy_x, gravity_scale, args.fs, args.hp_cut, args.lp_cut, args.eps, args.acc_indices, args.gyro_indices
    )
    print(f"Physics losses: {L_grav.item()}, {L_ag_combined.item()}, {L_jerk.item()}")
