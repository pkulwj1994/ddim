data:
    dataset: "LSUN"
    category: "bedroom"
    image_size: 256
    channels: 3
    logit_transform: false
    uniform_dequantization: false
    gaussian_dequantization: false
    random_flip: true
    rescaled: true
    num_workers: 32

model:
    type: "simple"
    in_channels: 3
    out_ch: 3
    ch: 128
    ch_mult: [1, 1, 2, 2, 4, 4]
    num_res_blocks: 2
    attn_resolutions: [16, ]
    dropout: 0.0
    var_type: fixedsmall
    ema_rate: 0.999
    ema: True
    resamp_with_conv: True

diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000

training:
    batch_size: 64
    n_epochs: 10000
    n_iters: 5000000
    snapshot_freq: 5000
    validation_freq: 2000

sampling:
    batch_size: 32
    last_only: True

optim:
    weight_decay: 0.000
    optimizer: "Adam"
    lr: 0.00002
    beta1: 0.9
    amsgrad: false
    eps: 0.00000001
