import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# 1. 模型加载
# =========================
model_path = "/data/clx/control-lora-v2-master/ckpt/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path).to("cuda")
model.eval()

# =========================
# 2. 数据路径
# =========================
image_dir = "/data/clx/data/lora_data/lvbag_data/lv_bag"

img_list = os.listdir(image_dir)

# =========================
# 3. 清洗函数（只做“去垃圾”，不改内容）
# =========================
def clean_caption(text: str) -> str:
    remove_list = [
        "a photography of",
        "a photo of",
        "an image of",
        "the image shows",
    ]

    text = text.lower().strip()

    for r in remove_list:
        text = text.replace(r, "")

    text = text.strip(" ,.")

    if len(text) == 0:
        text = "handbag"

    return text


# =========================
# 4. 逐图生成 caption + txt
# =========================
for img_name in img_list:
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    img_path = os.path.join(image_dir, img_name)

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"跳过损坏图片: {img_name}")
        continue

    # BLIP输入
    inputs = processor(image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30
        )

    caption = processor.decode(out[0], skip_special_tokens=True)

    # 清洗
    caption = clean_caption(caption)

    # 加 trigger token（核心）
    final_caption = f"<sksbag>, {caption}"

    # =========================
    # 5. 写入同名 txt
    # =========================
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(image_dir, txt_name)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_caption)

    print(f"已生成: {img_name} -> {txt_name}")

print("全部处理完成")