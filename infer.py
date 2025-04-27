import os
import torch
from PIL import Image
from modelscope import AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def predict(image_path, model, processor, resize_hw=(280, 280)):
    # 读取并Resize图像
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize(resize_hw)
    temp_save_path = "temp_resized_image.jpg"
    resized_image.save(temp_save_path)

    # 构建对话输入
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": temp_save_path},
            {"type": "text", "text": "消防安全风险等级是？"}
        ]
    }]

    # 文本预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to("cuda")

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    # 清理资源
    del inputs
    torch.cuda.empty_cache()
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)

    return output_text[0].strip()

def main():
    model_path = r"C:\Users\Administrator\.cache\modelscope\hub\models\qwen\Qwen2.5-VL-7B-Instruct"
    lora_checkpoint = r"D:\Aliyun\ELE_AI_XiaoFang\BLIP_GPT\QwenAPP\output\Qwen2.5-VL-7B\checkpoint-1200"
    image_folder = r"D:\Aliyun\ELE_AI_XiaoFang\data\A"
    output_txt_path = r"D:\Aliyun\ELE_AI_XiaoFang\data\dataset\predict_result.txt"

    # 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)

    # 加载主模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # 配置并加载LoRA权重
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    model = PeftModel.from_pretrained(model, model_id=lora_checkpoint, config=val_config).eval()

    # 遍历图片推理
    # 遍历图片推理
    image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_list = sorted(image_list, key=lambda x: x.lower())  # 保持原图像名称自然升序排列


    with open(output_txt_path, "w", encoding="utf-8", newline="\n") as f_out:
        for img_name in image_list:
            img_path = os.path.join(image_folder, img_name)
            try:
                risk_level = predict(img_path, model, processor)
                result_line = f"{img_name}\t{risk_level}\n"
                f_out.write(result_line)
                print(result_line.strip())
            except Exception as e:
                print(f"[Error] {img_name}: {e}")
                continue

    print(f"\n✅ 所有推理完成，结果保存在 {output_txt_path}")

if __name__ == "__main__":
    main()
