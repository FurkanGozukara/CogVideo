import argparse
import platform
import gradio as gr
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
import os
import random
import glob

# Default values
default_prompt = ""
default_image_path = "input.png"
default_num_videos = 1
default_num_inference_steps = 50
default_num_frames = 81
default_guidance_scale = 6
default_fps = 16
default_seed = 42
default_width = 768
default_height = 768

# Initialize the model and pipeline globally
pipe = None

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def setup_model(enable_quantization, enable_cpu_offload):
    global pipe
    if enable_quantization:
        quantization = int8_weight_only
        text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX1.5-5B-I2V", subfolder="text_encoder",
                                                      torch_dtype=torch.bfloat16)
        quantize_(text_encoder, quantization())

        transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX1.5-5B-I2V", subfolder="transformer",
                                                                  torch_dtype=torch.bfloat16)
        quantize_(transformer, quantization())

        vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX1.5-5B-I2V", subfolder="vae",
                                                     torch_dtype=torch.bfloat16)
        quantize_(vae, quantization())
    else:
        text_encoder = None
        transformer = None
        vae = None

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX1.5-5B-I2V",
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()

    return pipe

def get_next_video_number(output_dir):
    existing_videos = glob.glob(os.path.join(output_dir, "video_*.mp4"))
    if not existing_videos:
        return 1
    
    numbers = [int(os.path.basename(v).split('_')[1].split('.')[0]) for v in existing_videos]
    return max(numbers) + 1

def generate_video(
    prompt,
    image_path,
    num_videos,
    num_inference_steps,
    num_frames,
    guidance_scale,
    fps,
    seed,
    width,
    height,
    enable_vae_tiling,
    enable_vae_slicing,
    enable_quantization,
    enable_cpu_offload,
    use_random_seed
):
    global pipe
    try:
        if not prompt:
            raise gr.Error("Please enter a prompt.")
        if not image_path or not os.path.exists(image_path):
            raise gr.Error("Please provide a valid image path.")

        if pipe is None:
            pipe = setup_model(enable_quantization, enable_cpu_offload)
        elif enable_quantization != getattr(pipe, 'quantization_enabled', None) or enable_cpu_offload != getattr(pipe, 'cpu_offload_enabled', None):
            pipe = setup_model(enable_quantization, enable_cpu_offload)

        setattr(pipe, 'quantization_enabled', enable_quantization)
        setattr(pipe, 'cpu_offload_enabled', enable_cpu_offload)

        image = load_image(image_path)
        
        if use_random_seed or seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        if enable_vae_tiling:
            pipe.vae.enable_tiling()
        else:
            pipe.vae.disable_tiling()

        if enable_vae_slicing:
            pipe.vae.enable_slicing()
        else:
            pipe.vae.disable_slicing()

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        video_paths = []
        for i in range(num_videos):
            video = pipe(
                prompt=prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
            
            # Get the next available video number
            video_number = get_next_video_number(output_dir)
            video_filename = f"video_{video_number:04d}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            export_to_video(video, video_path, fps=fps)
            video_paths.append(video_path)

        return video_paths[0] if video_paths else None, seed

    except Exception as e:
        raise gr.Error(f"Error during video generation: {e}")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", help="Share the app using Gradio's share=True")
args = parser.parse_args()

# Gradio Interface
with gr.Blocks(title="CogVideoX Text-to-Video Generation") as demo:
    gr.Markdown("# CogVideoX1.5-5B-I2V SECourses APP V1 - https://www.patreon.com/posts/112848192")
    
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value=default_prompt,
            )
            generate_button = gr.Button("Generate Video")
            image_path = gr.Image(
                label="Input Image", type="filepath", height=512
            )
            num_videos = gr.Slider(
                label="Number of Videos",
                minimum=1,
                maximum=50,
                step=1,
                value=default_num_videos,
            )
            
        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video", format="mp4")
            btn_open_outputs = gr.Button("Open Outputs Folder")
            btn_open_outputs.click(fn=open_folder)
            with gr.Accordion("Advanced Options", open=True):
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Number of Inference Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=default_num_inference_steps,
                    )
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=1,
                        maximum=300,
                        step=1,
                        value=default_num_frames,
                    )
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=1536,
                        step=16,
                        value=default_width,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=1536,
                        step=16,
                        value=default_height,
                    )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=20,
                    step=0.5,
                    value=default_guidance_scale,
                )
                fps = gr.Slider(
                    label="Output FPS",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=default_fps,
                )
                with gr.Row():
                    use_random_seed = gr.Checkbox(
                        label="Use Random Seed", value=True
                    )
                    seed = gr.Number(
                        label="Seed", value=default_seed, precision=0
                    )
                with gr.Row():
                    enable_quantization = gr.Checkbox(
                        label="Enable Quantization", value=True
                    )
                    enable_cpu_offload = gr.Checkbox(
                        label="Enable Model CPU Offload", value=True
                    )
                with gr.Row():
                    enable_vae_tiling = gr.Checkbox(
                        label="Enable VAE Tiling", value=True
                    )
                    enable_vae_slicing = gr.Checkbox(
                        label="Enable VAE Slicing", value=True
                    )

    def update_seed_interactivity(use_random):
        return gr.update(interactive=not use_random)

    # Connect the use_random_seed checkbox to the update_seed_interactivity function
    use_random_seed.change(
        fn=update_seed_interactivity,
        inputs=use_random_seed,
        outputs=seed,
    )

    generate_button.click(
        fn=generate_video,
        inputs=[
            prompt,
            image_path,
            num_videos,
            num_inference_steps,
            num_frames,
            guidance_scale,
            fps,
            seed,
            width,
            height,
            enable_vae_tiling,
            enable_vae_slicing,
            enable_quantization,
            enable_cpu_offload,
            use_random_seed
        ],
        outputs=[video_output, seed],
    )

# Add some custom CSS for better styling
demo.css = """
    .gradio-container {max-width: 960px; margin: auto;}
    .gr-button {background-color: #007bff; color: white;}
    .gr-button:hover {background-color: #0056b3;}
    .gr-form {border: 1px solid #e0e0e0; padding: 15px; border-radius: 5px;}
    .gr-form > div {margin-bottom: 15px;}
"""

# Launch the interface
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=args.share)