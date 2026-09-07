import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional, List

@dataclass
class Options:
    ### tokenizer
    # coord discrete bins (also the number of basic tokens)
    discrete_bins: int = 256
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    start_epoch: int = 0

    ### lmm
    # freeze encoder
    freeze_encoder: bool = None
    # max sequence length (excluding BOS, EOS, and COND)
    max_seq_length: int = 10240
    # hidden size
    hidden_dim: int = 1024
    # intermediate mlp size
    intermediate_dim: Optional[int] = None
    # layers
    num_layers: int = 24
    # num head
    num_heads: int = 16
    # conditions
    cond_mode: Literal['none', 'text', 'image', 'image+depth', 'image+text', 'depth+image+text'] = 'text'
    # length of condition tokens
    num_cond_tokens: int = 77
    # text 77, image 257
    generate_mode: Literal['greedy', 'sample'] = 'sample'
    # num face condition
    use_num_face_cond: bool = False
    # number of face dropout ratio
    nof_dropout_ratio: float = 0.2
    img_size: int = 512
    pose_length: int = 30
    normalized_cameras: bool = True
    target_height: int = 512
    target_width: int = 512
    camera_token_size: int = 9
    ### dataset
    # max face length
    max_face_length: int = 1000
    # data set
    dataset: Literal['obj', 'objxl'] = 'obj'
    # num workers
    num_workers: int = 64
    # testset size
    testset_size: int = 1
    # decimate aug
    use_decimate_aug: bool = True
    # scale aug
    use_scale_aug: bool = True
    # dataset path
    path: str = "DataDoP/train"
    
    ### training
    # workspace
    
    exp_name: str = ''
    workspace: str = './workspace'
    # resume ckpt path
    resume: Optional[str] = None
    resume2: Optional[str] = None
    # resume step_ratio
    resume_step_ratio: float = 0
    # pos embd align
    align_posemb: Literal['left', 'right'] = 'right'
    # batch size (per-GPU)
    batch_size: int = 16
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 100
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: Literal['no', 'fp8', 'fp16', 'fp32'] = 'bf16'
    # learning rate
    lr: float = 1e-4
    # gradient checkpointing
    checkpointing: bool = True
    # random seed
    seed: int = 0
    # evaluate mode
    eval_mode: Literal['none', 'loss', 'generate'] = 'loss'
    # debug eval in train (skip training and only do evaluation)
    debug_eval: bool = False
    # lr warmup ratio
    warmup_ratio: float = 0.01
    # use wandb
    use_wandb: bool = False
    # model save
    save_epoch: int = 50
    
    ### testing
    # test image/point path
    test_path: Optional[str] = None
    # test resume tokens
    test_resume_tokens: Optional[str] = None
    # test repeat
    test_repeat: int = 1
    # test targeted num faces (can be a list)
    test_num_face: Tuple[int, ...] = (1000,)
    # test max seq len
    test_max_seq_length: Optional[int] = None
    
    text_key: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    text_path: Optional[str] = None
    image_path: Optional[str] = None
    depth_path: Optional[str] = None

    
# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['default'] = 'the default settings'
config_defaults['default'] = Options()

config_doc['ArAE'] = 'ArAE'
config_defaults['ArAE'] = Options(
    discrete_bins=256,
    use_num_face_cond=True,
    use_decimate_aug=True,    
    cond_mode='text',
    num_cond_tokens=77,
    freeze_encoder=False,
    max_face_length=4000,
    max_seq_length=40960,
    align_posemb='right',
    batch_size=16,
    hidden_dim=1024,
    num_heads=8,
    num_layers=12,
    # small 512-8-8
    # base 1024-8-12
    # large 1536-16-24
    gradient_accumulation_steps=1,
    lr=1e-5,
    warmup_ratio=0,
    num_epochs=1000,
    eval_mode='loss',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)