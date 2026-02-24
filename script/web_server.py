# import gradio as gr
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import os
# import random
# from PIL import ImageDraw
# from matplotlib import pyplot as plt
# from pathlib import Path

# IMG_SIZE = 224

# # 模型載入
# print("開始載入 VGG19 模型...")
# vgg_model = tf.keras.models.load_model("./model/vgg19_transfer_model.h5")
# print("已載入 VGG19 模型")

# print("開始載入 Cat Generator 模型...")
# cat_gen = tf.keras.models.load_model("./model/cat150.keras")
# print("已載入 Cat Generator 模型")

# print("開始載入 Dog Generator 模型...")
# dog_gen = tf.keras.models.load_model("./model/dog150.keras")
# print("已載入 Dog Generator 模型")

# # 預測與生成主功能
# def predict_and_generate(image):
    
#     if isinstance(image, dict) and "layers" in image:
#         # 來自 Sketchpad，取最後一層
#         last_layer = image["layers"][-1]
#         image = Image.fromarray(last_layer).convert("RGB")
#     elif isinstance(image, np.ndarray):
#         image = Image.fromarray(image).convert("RGB")
#     elif isinstance(image, Image.Image):
#         image = image.convert("RGB")
#     else:
#         return "輸入格式錯誤", None

#     # 儲存原圖做比對
#     image.save("./output/input_saved.png")

#     # 處理為 VGG19 輸入
#     image_rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
#     img_array = np.array(image_rgb).astype(np.float32)
#     input_tensor = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))

#     # 分類
#     preds = vgg_model.predict(input_tensor)[0]
#     label = np.argmax(preds)
#     confidence = preds[label] * 100

#     if label == 0:
#         label_text = "cat"
#         generator = cat_gen
#     elif label == 1:
#         label_text = "dog"
#         generator = dog_gen
#     else:
#         return "other (不生成)", image

#     # 生成圖（256x256）
#     image_resized = image.convert("RGB").resize((256, 256))
#     input_gen = np.array(image_resized).astype(np.float32) / 255.0
#     input_gen = input_gen.reshape((1, 256, 256, 3))
#     output = generator(input_gen, training=True)[0].numpy()
#     output = ((output + 1) / 2 * 255).astype(np.uint8) 
#     output_image = Image.fromarray(output)

#     return f"{label_text} ({confidence:.2f}%)", output_image

# def load_random_image():
#     folder_path = "./dataset/testedge/"
#     images = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
#     if not images:
#         return None

#     random_path = random.choice(images)
#     image = Image.open(random_path).convert("L")  # 灰階

#     # 保持比例縮放
#     image.thumbnail((256, 256), Image.Resampling.LANCZOS)

#     # 建立 256x256 的白底畫布,並將圖片貼上去(置中)
#     canvas = Image.new("L", (256, 256), color=255)
#     offset = ((256 - image.width) // 2, (256 - image.height) // 2)
#     canvas.paste(image, offset)

#     image_array = np.array(canvas).astype(np.uint8)

#     # 將圖片直接放在 composite,並提供一個透明/空白的繪製層
#     return {
#         "background": None,  # 背景圖
#         "layers": [image_array],  # 空的圖層列表,讓用戶可以繪製
#         "composite": None  # 讓系統自動合成
#     }

# with gr.Blocks() as demo:
#     gr.Markdown("# 🐾 AI Cat/Dog Generator")
#     gr.Markdown("選擇上傳圖片或進行手繪，AI 會進行分類與風格生成")

#     with gr.Tabs():
#         with gr.Tab("📤 上傳圖片"):
#             with gr.Row():
#                 upload_input = gr.Image(
#                     type="pil",
#                     label="上傳圖片",
#                 )
#             upload_output_label = gr.Textbox(label="分類結果")
#             upload_output_image = gr.Image(label="生成圖像")
#             upload_submit_btn = gr.Button("✨ 進行分類與生成")
#             upload_submit_btn.click(predict_and_generate, inputs=upload_input, outputs=[upload_output_label, upload_output_image])

#         with gr.Tab("✏️ 手繪模式"):
#             with gr.Row():
#                 draw_input = gr.Sketchpad(
#                     label="手繪或貼上隨機圖片",
#                     brush=gr.Brush(colors=["black", "gray"]),
#                     canvas_size=(256, 256),
#                     type="numpy"
#                 )
#             draw_output_label = gr.Textbox(label="分類結果")
#             draw_output_image = gr.Image(label="生成圖像")
#             with gr.Row():
#                 draw_submit_btn = gr.Button("✨ 進行分類與生成")
#                 draw_clear_btn = gr.Button("🧹 清除")
#                 draw_random_btn = gr.Button("🎲 載入隨機圖片")

#             draw_submit_btn.click(predict_and_generate, inputs=draw_input, outputs=[draw_output_label, draw_output_image])
#             draw_clear_btn.click(lambda: None, None, draw_input, queue=False)
#             draw_random_btn.click(load_random_image, outputs=draw_input)

# demo.launch(server_name="0.0.0.0", server_port=7860)


import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
from pathlib import Path

IMG_SIZE = 224

# 模型載入
print("開始載入 VGG19 模型...")
vgg_model = tf.keras.models.load_model("./model/vgg19_transfer_model.h5")
print("已載入 VGG19 模型")

print("開始載入 Cat Generator 模型...")
cat_gen = tf.keras.models.load_model("./model/cat150.keras")
print("已載入 Cat Generator 模型")

print("開始載入 Dog Generator 模型...")
dog_gen = tf.keras.models.load_model("./model/dog150.keras")
print("已載入 Dog Generator 模型")


def preprocess_image_for_generator(image):
    """
    統一的圖片預處理函數 - 確保與 Pix2Pix 訓練時完全一致
    
    Args:
        image: PIL Image 或 numpy array
    
    Returns:
        preprocessed: (1, 256, 256, 3) 的 numpy array, 值域 [-1, 1]
    """
    # 1. 轉換為 PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 2. 轉換為灰階 (重要!)
    image_gray = image.convert("L")
    
    # 3. Resize 到 256x256 (使用高品質重採樣)
    image_resized = image_gray.resize((256, 256), Image.Resampling.LANCZOS)
    
    # 4. 轉換為 numpy array
    img_array = np.array(image_resized, dtype=np.float32)
    
    # 5. 轉換為 3 通道 (Pix2Pix 通常需要 RGB 格式)
    img_array = np.stack([img_array, img_array, img_array], axis=-1)
    
    # 6. 正規化到 [-1, 1] (這是 Pix2Pix 的標準範圍!)
    img_array = (img_array / 127.5) - 1.0
    
    # 7. 加上 batch 維度
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_image_for_vgg(image):
    """
    VGG19 分類器的預處理
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(image_rgb, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_and_generate(image):
    """
    主要的預測與生成函數
    """
    # 處理 Sketchpad 輸入
    if isinstance(image, dict) and "composite" in image:
        # 使用 composite (已合成的最終圖像)
        if image["composite"] is not None:
            image = Image.fromarray(image["composite"])
        elif image["layers"] and len(image["layers"]) > 0:
            # 如果沒有 composite,使用最後一層
            image = Image.fromarray(image["layers"][-1])
        else:
            return "請繪製或上傳圖片", None
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        return "輸入格式錯誤", None
    
    # 轉換為 RGB (處理 RGBA 等格式)
    image = image.convert("RGB")
    
    # === 1. VGG19 分類 ===
    vgg_input = preprocess_image_for_vgg(image)
    preds = vgg_model.predict(vgg_input, verbose=0)[0]
    label = np.argmax(preds)
    confidence = preds[label] * 100
    
    if label == 0:
        label_text = "cat"
        generator = cat_gen
    elif label == 1:
        label_text = "dog"
        generator = dog_gen
    else:
        # return "other (不生成)", image
        label_text = "other"
        generator = cat_gen # if random() % 2 == 0 else dog_gen
    
    # === 2. Pix2Pix 生成 ===
    gen_input = preprocess_image_for_generator(image)
    
    # 儲存預處理後的輸入 (用於調試)
    debug_img = ((gen_input[0] + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(debug_img).save("./output/preprocessed_input.png")
    
    # 生成圖像
    output = generator(gen_input, training=True)[0].numpy()
    
    # 反正規化: [-1, 1] -> [0, 255]
    output = ((output + 1) / 2 * 255).astype(np.uint8)
    output = np.clip(output, 0, 255)
    
    output_image = Image.fromarray(output)
    output_image.save("./output/generated_output.png")
    
    return f"{label_text} ({confidence:.2f}%)", output_image


def load_random_image():
    """
    載入隨機測試圖片
    """
    folder_path = "./dataset/testedge/"
    images = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
    
    if not images:
        print("找不到測試圖片!")
        return None
    
    random_path = random.choice(images)
    image = Image.open(random_path).convert("L")
    
    # 保持比例縮放
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    
    # 建立 256x256 白底畫布
    canvas = Image.new("L", (256, 256), color=255)
    offset = ((256 - image.width) // 2, (256 - image.height) // 2)
    canvas.paste(image, offset)
    
    # 轉換為 RGB 格式給 Sketchpad
    canvas_rgb = canvas.convert("RGB")
    image_array = np.array(canvas_rgb)
    
    return image_array


# === Gradio 介面 ===
with gr.Blocks(title="AI Cat/Dog Generator") as demo:
    gr.Markdown("# 🐾 AI Cat/Dog Generator")
    gr.Markdown("上傳圖片或手繪邊緣圖,AI 會自動生成寫實的貓/狗圖像")
    gr.Markdown("⚠️ **注意**: 請使用灰階邊緣圖以獲得最佳效果")

    with gr.Tabs():
        # Tab 1: 上傳圖片
        with gr.Tab("📤 上傳圖片"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        type="pil",
                        label="上傳圖片 (建議使用灰階邊緣圖)",
                    )
                    upload_submit_btn = gr.Button("✨ 進行分類與生成", variant="primary")
                
                with gr.Column():
                    upload_output_label = gr.Textbox(label="分類結果")
                    upload_output_image = gr.Image(label="生成圖像")
            
            upload_submit_btn.click(
                predict_and_generate, 
                inputs=upload_input, 
                outputs=[upload_output_label, upload_output_image]
            )

        # Tab 2: 手繪模式
        with gr.Tab("✏️ 手繪模式"):
            with gr.Row():
                with gr.Column():
                    draw_input = gr.Sketchpad(
                        label="手繪或載入隨機圖片",
                        brush=gr.Brush(
                            colors=["#000000", "#808080", "#FFFFFF"],
                            default_size=3
                        ),
                        canvas_size=(256, 256),
                        type="numpy"
                    )
                    with gr.Row():
                        draw_submit_btn = gr.Button("✨ 進行分類與生成", variant="primary")
                        draw_clear_btn = gr.Button("🧹 清除")
                        draw_random_btn = gr.Button("🎲 載入隨機圖片")
                
                with gr.Column():
                    draw_output_label = gr.Textbox(label="分類結果")
                    draw_output_image = gr.Image(label="生成圖像")
            
            draw_submit_btn.click(
                predict_and_generate, 
                inputs=draw_input, 
                outputs=[draw_output_label, draw_output_image]
            )
            draw_clear_btn.click(lambda: None, None, draw_input)
            draw_random_btn.click(load_random_image, outputs=draw_input)

    demo.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        show_error=True
    )
    
    # local host run : 
    # /home/handsomeguy/anaconda3/envs/py310/bin/python /mnt/c/Users/jone9/Documents/Code_Project/college/junior/topic/script/web_server.py
    
    # link to docker : 
    # docker run -d --name topic-container -p 7860:7860 topic  
    # /snap/bin/ngrok http 7860