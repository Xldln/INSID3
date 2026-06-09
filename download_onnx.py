import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download


os.mkdir('pretrain/onnx_weights', exist_ok=True)



snapshot_download(
    'onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX',
    local_dir='pretrain/onnx_weights',
    local_dir_use_symlinks=False,
    force_download=True,
    resume_download=False,
)
