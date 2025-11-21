import torch

from model.deformable_detr import DeformableDetrConfig
from model.egtr import DetrForSceneGraphGeneration

# 1. 사용할 디바이스 선택 (M1이면 보통 MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    device_name = "MPS"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device("cpu")
    device_name = "CPU"

print(f"[INFO] Using device: {DEVICE} ({device_name})")

# 2. Deformable DETR 기반 EGTR 모델 불러오기
#    - pretrained 백본: SenseTime/deformable-detr (HuggingFace)
pretrained_name = "SenseTime/deformable-detr"

print(f"[INFO] Loading config from {pretrained_name} ...")
config = DeformableDetrConfig.from_pretrained(pretrained_name)

# 수식 OCR용이 아니라 그냥 sanity check라서
# 대충 적당한 값으로 num_labels, num_rel_labels 만 설정
config.num_labels = 50          # object class 개수 (dummy)
config.num_rel_labels = 10      # relation class 개수 (dummy)
config.num_queries = 100        # query 수 (조금 줄여서 가볍게)

# EGTR가 기대하는 추가 config 필드들 (train_egtr.py에서 쓰는 것들 흉내)
# 없으면 AttributeError 나니까, 없을 경우에만 기본값으로 채워준다.
if not hasattr(config, "use_freq_bias"):
    config.use_freq_bias = False

if not hasattr(config, "use_log_softmax"):
    config.use_log_softmax = False

if not hasattr(config, "filter_duplicate_rels"):
    config.filter_duplicate_rels = False

if not hasattr(config, "filter_multiple_rels"):
    config.filter_multiple_rels = False

# 혹시 relation 쪽에서 class agnostic 옵션 참조할 수도 있어서
if not hasattr(config, "rel_class_agnostic"):
    config.rel_class_agnostic = False

if not hasattr(config, "logit_adjustment"):
    config.logit_adjustment = False
if not hasattr(config, "logit_adj_tau"):
    config.logit_adj_tau = 1.0


print("[INFO] Creating DetrForSceneGraphGeneration model ...")
# fg_matrix는 지금은 안 쓰니까 None으로 둠
model, _ = DetrForSceneGraphGeneration.from_pretrained(
    pretrained_name,
    config=config,
    fg_matrix=None,
    ignore_mismatched_sizes=True,
    output_loading_info=True,
    device_map=None,   # M1이라 device_map은 직접 안 씀
)

model.to(DEVICE)
model.eval()

# 3. 더미 입력 만들기 (batch_size=1, 3채널, 224x224 이미지)
batch_size = 1
height = 224
width = 224

print(f"[INFO] Creating dummy inputs: batch={batch_size}, size={height}x{width}")
pixel_values = torch.rand(batch_size, 3, height, width, device=DEVICE)
pixel_mask = torch.ones(batch_size, height, width, dtype=torch.long, device=DEVICE)

# 4. Forward 한 번 태워보기
with torch.no_grad():
    print("[INFO] Running forward pass ...")
    outputs = model(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        output_attentions=False,
        output_attention_states=True,
        output_hidden_states=True,
    )

print("[OK] Forward pass finished without error!")
print(" - logits shape:", outputs.logits.shape)
print(" - pred_boxes shape:", outputs.pred_boxes.shape)
print(" - pred_rel shape:", outputs.pred_rel.shape)

