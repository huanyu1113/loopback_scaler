import math
import json
import bisect
import gradio as gr
import modules.scripts as scripts
from modules import deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state

class Script(scripts.Script):
    def title(self):
        return "Loopback Dynamic Prompts"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        loops = gr.Slider(minimum=1, maximum=4000, step=1, 
                         label='Loops', value=1200,
                         elem_id=self.elem_id("loops"))
        
        final_denoising = gr.Slider(minimum=0, maximum=1, step=0.01,
                                   label='Final Denoising Strength', value=0.5,
                                   elem_id=self.elem_id("final_denoising"))
        
        denoise_curve = gr.Dropdown(label="Denoising Curve",
                                   choices=["Aggressive", "Linear", "Lazy"], 
                                   value="Linear")
        
        append_mode = gr.Dropdown(label="Auto Append Prompt",
                                 choices=["None", "CLIP", "DeepBooru"],
                                 value="None")

        return [loops, final_denoising, denoise_curve, append_mode]

    def run(self, p, loops, final_denoising, denoise_curve, append_mode):
        processing.fix_seed(p)
        
        # 保存原始参数
        original_prompt = p.prompt
        original_images = p.init_images
        original_denoise = p.denoising_strength
        original_batch = p.batch_size
        original_inpaint = p.inpainting_fill
        
        # 解析JSON提示词
        prompt_map = {}
        sorted_thresholds = []
        try:
            if original_prompt.strip().startswith("{"):
                prompt_map = json.loads(original_prompt)
                # 转换并排序阈值
                sorted_thresholds = sorted(int(k) for k in prompt_map.keys() if k.isdigit())
                # 转换回字符串字典
                prompt_map = {str(k): v for k, v in prompt_map.items()}
        except Exception as e:
            print(f"Prompt parsing error: {str(e)}")

        # 初始化参数
        p.extra_generation_params = {
            "Final Denoising": final_denoising,
            "Denoise Curve": denoise_curve
        }
        
        p.batch_size = 1
        p.n_iter = 1
        state.job_count = loops * p.n_iter

        # 准备颜色修正
        init_color_correction = None
        if opts.img2img_color_correction:
            init_color_correction = processing.setup_color_correction(original_images[0])

        all_images = []
        history_images = []
        initial_seed = None
        initial_info = None

        def calculate_strength(current_loop):
            progress = current_loop / max(loops-1, 1)
            
            if denoise_curve == "Aggressive":
                factor = math.sin(progress * math.pi / 2)
            elif denoise_curve == "Lazy":
                factor = 1 - math.cos(progress * math.pi / 2)
            else:  # Linear
                factor = progress
                
            return original_denoise + (final_denoising - original_denoise) * factor

        # 主处理循环
        for batch in range(p.n_iter):
            if state.interrupted: break
            
            p.init_images = original_images
            current_image = None
            
            for i in range(loops):
                if state.interrupted or state.skipped: break
                
                # 动态提示词选择
                current_loop = i + 1
                selected_prompt = original_prompt  # 默认
                
                if sorted_thresholds:
                    # 使用bisect查找适用的阈值
                    idx = bisect.bisect_right(sorted_thresholds, current_loop) - 1
                    if idx >= 0:
                        selected_key = str(sorted_thresholds[idx])
                        selected_prompt = prompt_map.get(selected_key, original_prompt)

                # 自动附加提示词
                if append_mode != "None" and current_image is not None:
                    if append_mode == "CLIP":
                        tags = shared.interrogator.interrogate(current_image)
                    else:
                        tags = deepbooru.model.tag(current_image)
                    
                    if tags:
                        selected_prompt += ", " + tags if selected_prompt else tags

                # 更新参数
                p.prompt = selected_prompt
                p.denoising_strength = calculate_strength(i)
                
                if init_color_correction:
                    p.color_corrections = [init_color_correction]

                # 执行生成
                state.job = f"Batch {batch+1}/{p.n_iter}, Loop {current_loop}/{loops}"
                processed = processing.process_images(p)

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                if processed.images:
                    current_image = processed.images[0]
                    p.init_images = [current_image]
                    p.inpainting_fill = 1  # 保持原始内容
                    
                    if p.n_iter == 1:
                        history_images.append(current_image)
                    all_images.append(current_image)

                p.seed = -1  # 强制随机种子

            # 批次处理完成
            if p.n_iter > 1 and current_image:
                history_images.append(current_image)
                all_images.append(current_image)

        # 生成网格图
        final_images = []
        if len(history_images) > 1:
            grid = images.image_grid(history_images, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, 
                                 p.prompt, opts.grid_format, info=initial_info,
                                 grid=True, p=p)
            if opts.return_grid:
                final_images.append(grid)
        
        final_images += all_images

        # 恢复原始参数
        p.inpainting_fill = original_inpaint
        p.prompt = original_prompt

        return Processed(p, final_images, initial_seed, initial_info)