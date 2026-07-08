import os
import json
import argparse
from glob import glob

VALID_EXT = [".png", ".jpg", ".jpeg", ".webp"]

def is_image(file):
    return os.path.splitext(file)[-1].lower() in VALID_EXT

def load_caption(txt_path):
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def load_dataset(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(
    target_dir,
    output_file=None,
    condition_dir=None,
    prompt_single=None,
):
    # ===== 获取图片 =====
    target_files = sorted(
        [f for f in glob(os.path.join(target_dir, "*")) if is_image(f)]
    )

    print(f"找到 {len(target_files)} 张图片")

    valid_count = 0

    if not output_file:
        output_file = os.path.join(os.path.dirname(target_dir),'train.jsonl')

    with open(output_file, "w", encoding="utf-8") as f:
        for i, tgt_file in enumerate(target_files):
            if prompt_single:
                prompt = prompt_single

            else:
                txt_file = os.path.splitext(tgt_file)[0] + ".txt"
                prompt = load_caption(txt_file)

                if prompt is None:
                    print(f"缺少caption，跳过: {tgt_file}")
                    continue

            cond_file = None
            if condition_dir:
                cond_file = os.path.join(condition_dir,os.path.basename(tgt_file))
                assert os.path.exists(cond_file)
            data = {
                "target": tgt_file,
                "condition": cond_file,
                "prompt": prompt,
            }

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"有效样本: {valid_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 JSONL 数据集")
    parser.add_argument("--target_dir", type=str, default='/data/clx/data/道路后处理/road_pro')
    parser.add_argument("--condition_dir", type=str, default= '/data/clx/data/道路后处理/road_png', help="Image to Image")
    parser.add_argument("--output_file", type=str, default= None, help="输出路径")
    parser.add_argument("--prompt_single",type=str,default= '道路二值分割掩膜，平滑连续的边界，没有锯齿状边缘，没有孔洞。', help="统一 prompt")

    args = parser.parse_args()
    build_dataset(
        target_dir=args.target_dir,
        condition_dir=args.condition_dir,
        output_file=args.output_file,
        prompt_single=args.prompt_single,
    )

