import gradio as gr
import spaces
css = '''
.gradio-container {width: 85% !important}
'''
from animatediff.utils.util import save_videos_grid

import random
from infer import load_model
MAX_SEED=10000
import uuid
from insightface.app import FaceAnalysis
import os
import os
import cv2
from diffusers.utils import load_image
from insightface.utils import face_align
from PIL import Image
import torch
import argparse
# From command line read command adaface_ckpt_path
parser = argparse.ArgumentParser()
parser.add_argument('--adaface_ckpt_path', type=str, 
                    default='models/adaface/subjects-celebrity2024-05-16T17-22-46_zero3-ada-30000.pt')
# Don't use 'sd15' for base_model_type; it just generates messy videos.
parser.add_argument('--base_model_type', type=str, default='sar')
parser.add_argument('--adaface_base_model_type', type=str, default='sar')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--ip', type=str, default="0.0.0.0")
args = parser.parse_args()

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# model = load_model()
# This FaceAnalysis uses a different model from what AdaFace uses, but it's fine.
# This is just to crop the face areas from the uploaded images.
app = FaceAnalysis(name="buffalo_l", root='models/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))
device = "cuda" if args.gpu is None else f"cuda:{args.gpu}"

id_animator, adaface = load_model(base_model_type=args.base_model_type, 
                                  adaface_base_model_type=args.adaface_base_model_type,
                                  adaface_ckpt_path=args.adaface_ckpt_path, 
                                  device=device)
basedir     = os.getcwd()
savedir     = os.path.join(basedir,'samples')
os.makedirs(savedir, exist_ok=True)

#print(f"### Cleaning cached examples ...")
#os.system(f"rm -rf gradio_cached_examples/")

def swap_to_gallery(images):
    # Update uploaded_files_gallery, show files, hide clear_button_column
    # Or:
    # Update uploaded_init_img_gallery, show init_img_files, hide init_clear_button_column
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(value=images, visible=False)

def remove_back_to_files():
    # Hide uploaded_files_gallery,    show clear_button_column,      hide files,           reset init_img_selected_idx
    # Or:
    # Hide uploaded_init_img_gallery, hide init_clear_button_column, show init_img_files,  reset init_img_selected_idx
    return gr.update(visible=False), gr.update(visible=False), gr.update(value=None, visible=True), gr.update(value="0")

def get_clicked_image(data: gr.SelectData):
    return data.index
    
@spaces.GPU
def gen_init_images(uploaded_image_paths, prompt, adaface_id_cfg_scale, out_image_count=3):
    if uploaded_image_paths is None:
        print("No image uploaded")
        return None, None, None
    # uploaded_image_paths is a list of tuples:
    # [('/tmp/gradio/249981e66a7c665aaaf1c7eaeb24949af4366c88/jensen huang.jpg', None)]
    # Extract the file paths.
    uploaded_image_paths = [path[0] for path in uploaded_image_paths]
    # gen_init_images() uses a larger adaface_id_cfg_scale to generate more authentic faces.
    adaface_id_cfg_scale_ = min(6, adaface_id_cfg_scale * 2)
    adaface_subj_embs = \
        adaface.generate_adaface_embeddings(image_folder=None, image_paths=uploaded_image_paths,
                                            out_id_embs_scale=adaface_id_cfg_scale_, update_text_encoder=True)
    
    if adaface_subj_embs is None:
        raise gr.Error(f"Failed to detect any faces! Please try with other images")
        
    # Generate two images each time for the user to select from.
    noise = torch.randn(out_image_count, 3, 512, 512)
    # samples: A list of PIL Image instances.
    samples = adaface(noise, prompt, out_image_count=out_image_count, verbose=True)

    face_paths = []
    for sample in samples:        
        random_name = str(uuid.uuid4())
        face_path = os.path.join(savedir, f"{random_name}.jpg")
        face_paths.append(face_path)
        sample.save(face_path)
        print(f"Generated init image: {face_path}")

    # Update uploaded_init_img_gallery, update and hide init_img_files, hide init_clear_button_column
    return gr.update(value=face_paths, visible=True), gr.update(value=face_paths, visible=False), gr.update(visible=True)

@spaces.GPU(duration=90)
def generate_image(image_container, uploaded_image_paths, init_img_file_paths, init_img_selected_idx, 
                   init_image_strength, init_image_final_weight,
                   prompt, negative_prompt, num_steps, video_length, guidance_scale, seed, attn_scale, image_embed_scale,
                   is_adaface_enabled, adaface_ckpt_path, adaface_id_cfg_scale, adaface_power_scale, 
                   adaface_anneal_steps, progress=gr.Progress(track_tqdm=True)):

    prompt = prompt + " 8k uhd, high quality"
    if " shot" not in prompt:
        prompt = prompt + ", medium shot"
        
    prompt_img_lists=[]
    for path in uploaded_image_paths:
        img = cv2.imread(path)
        faces = app.get(img)
        face_roi = face_align.norm_crop(img, faces[0]['kps'], 112)
        random_name = str(uuid.uuid4())
        face_path = os.path.join(savedir, f"{random_name}.jpg")
        cv2.imwrite(face_path, face_roi)
        # prompt_img_lists is a list of PIL images.
        prompt_img_lists.append(load_image(face_path).resize((224,224)))

    if adaface is None or not is_adaface_enabled:
        adaface_prompt_embeds = None
    else:
        if adaface_ckpt_path != args.adaface_ckpt_path:
            # Reload the embedding manager
            adaface.load_subj_basis_generator(adaface_ckpt_path)

        adaface.generate_adaface_embeddings(image_folder=None, image_paths=uploaded_image_paths,
                                            out_id_embs_scale=adaface_id_cfg_scale, update_text_encoder=True)
        # adaface_prompt_embeds: [1, 77, 768].
        adaface_prompt_embeds, _ = adaface.encode_prompt(prompt)

    # init_img_file_paths is a list of image paths. If not chose, init_img_file_paths is None.
    if init_img_file_paths is not None:
        init_img_selected_idx = int(init_img_selected_idx)
        init_img_file_path = init_img_file_paths[init_img_selected_idx]
        init_image = cv2.imread(init_img_file_path)
        init_image = cv2.resize(init_image, (512, 512))
        init_image = Image.fromarray(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))
        print(f"init_image: {init_img_file_path}")
    else:
        init_image = None

    sample = id_animator.generate(prompt_img_lists, 
                                  init_image      = init_image,
                                  init_image_strength = (init_image_strength, init_image_final_weight),
                                  prompt = prompt,
                                  negative_prompt = negative_prompt,
                                  adaface_embeds  = adaface_prompt_embeds,
                                  # adaface_scale is not so useful, and when it's set >= 2, weird artifacts appear. 
                                  # Here it's limited to 0.7~1.3.
                                  adaface_scale       = adaface_power_scale,
                                  num_inference_steps = num_steps,
                                  adaface_anneal_steps = adaface_anneal_steps,
                                  seed=seed,
                                  guidance_scale      = guidance_scale,
                                  width               = 512,
                                  height              = 512,
                                  video_length        = video_length,
                                  attn_scale          = attn_scale,
                                  image_embed_scale   = image_embed_scale,
                                )
    
    save_sample_path = os.path.join(savedir, f"{random_name}.mp4")
    save_videos_grid(sample, save_sample_path)
    return save_sample_path

def validate_prompt(prompt):
    if not prompt:
        raise gr.Error("Prompt cannot be blank")

examples = [
    [
        "demo/ann.png",
        ["demo/ann.png" ],
        "A young girl with a passion for reading, curled up with a book in a cozy nook near a window",
        "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,",
        30,
        8, 8290,1,16
    ],
    [
        "demo/lecun.png",
        ["demo/lecun.png" ],
        "Iron Man soars through the clouds, his repulsors blazing",
        "worst quality, low quality, jpeg artifacts, ugly, duplicate, blurry, long neck",
        30,
        8, 4993,0.7,16
    ],
    [
        "demo/mix.png",
        ["demo/lecun.png","demo/ann.png"],
        "A musician playing a guitar, fingers deftly moving across the strings, producing a soulful melody",
        "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        30,
        8, 1897,0.9,16
    ],
    [
        "demo/zendaya.png",
        ["demo/zendaya.png" ],
        "A woman on a serene beach at sunset, the sky ablaze with hues of orange and purple.",
        "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        30,
        8, 5992,1,16
    ],
    [
        "demo/qianlong.png",
        ["demo/qianlong.png" ],
        "A chef in a white apron, complete with a toqueblanche, garnishing a gourmet dish",
        "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream",
        30,
        8, 1844,0.8,16
    ],
    [
        "demo/augustus.png",
        ["demo/augustus.png" ],
        "A man with dyed pink and purple hair, styledin a high ponytail",
        "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        30,
        8, 870,0.7,16
    ]
]

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # AdaFace-Animate: Zero-Shot Subject-Driven Video Generation for Humans
        """
    )
    gr.Markdown(
        """
<b>Official demo</b> for our NeurIPS 2024 submission <b>AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization</b>.<br>

❗️**Tips**❗️
- You can upload one or more subject images for generating ID-specific video.
- Try different parameter combinations for the best generation quality.
- Usage explanations and demos: [Readme](https://huggingface.co/spaces/adaface-neurips/adaface-animate/blob/main/README2.md).
- AdaFace Text-to-Image: <a href="https://huggingface.co/spaces/adaface-neurips/adaface" style="display: inline-flex; align-items: center;">
  AdaFace 
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces" style="margin-left: 5px;">
  </a>

**TODO:**
- ControlNet integration.
        """
    )

    with gr.Row():
        with gr.Column():
            files = gr.File(
                        label="Drag / Select 1 or more photos of a person's face",
                        file_types=["image"],
                        file_count="multiple"
                    )
            image_container = gr.Image(label="image container", sources="upload", type="numpy", height=256, visible=False)
            uploaded_files_gallery = gr.Gallery(label="Subject images", visible=False, columns=3, rows=1, height=300)
            with gr.Column(visible=False) as clear_button_column:
                remove_and_reupload = gr.ClearButton(value="Remove and upload subject images", components=files, size="sm")

            init_img_files = gr.File(
                            label="[Optional] Select 1 image for initialization, or generate 3 images with the button below and select 1",
                            file_types=["image"],
                            file_count="multiple"
                    )
            init_img_container = gr.Image(label="init image container", sources="upload", type="numpy", height=256, visible=False)
            # Although there's only one image, we still use columns=3, to scale down the image size.
            # Otherwise it will occupy the full width, and the gallery won't show the whole image.
            uploaded_init_img_gallery = gr.Gallery(label="Init image", visible=False, columns=3, rows=1, height=200)
            # placeholder is just hint, not the real value. So we use "value='0'" instead of "placeholder='0'".
            init_img_selected_idx = gr.Textbox(label="Selected init image index", value="0", visible=False)

            init_image_strength = gr.Slider(
                    label="Init Image Strength",
                    info="How much the init image should influence each frame. 0: no influence (scenes are more dynamic), 3: strongest influence (scenes are more static).",
                    minimum=0,
                    maximum=3,
                    step=0.25,
                    value=1.5,
                )
            init_image_final_weight = gr.Slider(
                    label="Final Weight of the Init Image",
                    info="How much the init image should influence the end of the video",
                    minimum=0,
                    maximum=0.25,
                    step=0.025,
                    value=0.1,
                )

            with gr.Column(visible=False) as init_clear_button_column:
                remove_init_and_reupload = gr.ClearButton(value="Remove and upload new init image", components=init_img_files, size="sm")
            with gr.Column(visible=True) as init_gen_button_column:
                gen_init = gr.Button(value="Generate 3 new init images")

            prompt = gr.Dropdown(label="Prompt",
                       info="Try something like 'man/woman walking on the beach'. If the face is not in focus, try adding 'face portrait of' at the beginning.",
                       value=None,
                       allow_custom_value=True,
                       filterable=False,
                       choices=[
                            "woman ((best quality)), ((masterpiece)), ((realistic)), long highlighted hair, futuristic silver armor suit, confident stance, high-resolution, living room, smiling, head tilted, perfect smooth skin",
                            "woman walking on the beach, sunset, orange sky, eye level shot",
                            "woman in a white apron and chef hat, garnishing a gourmet dish, full body view, long shot",
                            "woman dancing pose among folks in a park, waving hands",
                            "woman in iron man costume flying pose, the sky ablaze with hues of orange and purple, full body view, long shot",
                            "woman jedi wielding a lightsaber, star wars, full body view, eye level shot",
                            "woman playing guitar on a boat, ocean waves",
                            "woman with a passion for reading, curled up with a book in a cozy nook near a window",
                            "woman running pose in a park, eye level shot",
                            "woman in superman costume flying pose, the sky ablaze with hues of orange and purple, full body view, long shot"
                       ])
            
            image_embed_scale = gr.Slider(
                    label="ID-Animator Image Embedding Scale",
                    info="The scale of the ID-Animator image embedding (influencing coarse facial features and poses)",
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=0.8,
                )
            attn_scale = gr.Slider(
                    label="Attention Processor Scale",
                    info="The scale of the ID embeddings on the attention (the higher, the more focus on the face, less on the background)" ,
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=0.8,
                )
            adaface_id_cfg_scale = gr.Slider(
                    label="AdaFace CFG Scale",
                    info="The CFG scale of the AdaFace ID embeddings (influencing fine facial features)",
                    minimum=0.5,
                    maximum=6,
                    step=0.25,
                    value=1.5,
                )

            submit = gr.Button("Generate Video")

            with gr.Accordion(open=False, label="Advanced Options"):
                video_length = gr.Slider(
                    label="video_length",
                    info="Do not change, otherwise the video will be messy",
                    minimum=16,
                    maximum=21,
                    step=1,
                    value=16,
                    interactive=False,
                )
                is_adaface_enabled = gr.Checkbox(label="Enable AdaFace", 
                                                 info="Enable AdaFace for better face details. If unchecked, it falls back to ID-Animator (https://huggingface.co/spaces/ID-Animator/ID-Animator).",
                                                 value=True)
                adaface_ckpt_path = gr.Textbox(
                    label="AdaFace checkpoint path", 
                    placeholder=args.adaface_ckpt_path,
                    value=args.adaface_ckpt_path,
                )

                adaface_power_scale = gr.Slider(
                        label="AdaFace Embedding Power Scale",
                        info="Increase this scale slightly only if the face is defocused or the face details are not clear",
                        minimum=0.8,
                        maximum=1.2,
                        step=0.1,
                        value=1,
                    )
                             
                # adaface_anneal_steps is no longer necessary, but we keep it here for future use.
                adaface_anneal_steps = gr.Slider(
                    label="AdaFace Anneal Steps",
                    minimum=0,
                    maximum=2,
                    step=1,
                    value=0,
                    visible=False,
                )
                                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="face portrait, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, bare breasts, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, long neck, UnrealisticDream",
                )
                num_steps = gr.Slider( 
                    label="Number of sampling steps",
                    minimum=25,
                    maximum=100,
                    step=1,
                    value=40,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale (usually you don't need to change)",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=4,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=985,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False, info="Uncheck for reproducible results")

        with gr.Column():
            result_video = gr.Video(label="Generated Animation", interactive=False)
        
        files.upload(fn=swap_to_gallery, inputs=files,     outputs=[uploaded_files_gallery, clear_button_column, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files_gallery, clear_button_column, files, init_img_selected_idx])

        init_img_files.upload(fn=swap_to_gallery, inputs=init_img_files, outputs=[uploaded_init_img_gallery, init_clear_button_column, init_img_files])
        remove_init_and_reupload.click(fn=remove_back_to_files,        outputs=[uploaded_init_img_gallery, init_clear_button_column, 
                                                                                init_img_files, init_img_selected_idx])
        gen_init.click(fn=gen_init_images, inputs=[uploaded_files_gallery, prompt, adaface_id_cfg_scale], 
                       outputs=[uploaded_init_img_gallery, init_img_files, init_clear_button_column])
        uploaded_init_img_gallery.select(fn=get_clicked_image, inputs=None, outputs=init_img_selected_idx)

        submit.click(fn=validate_prompt,
                     inputs=[prompt],outputs=None).success(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
                 fn=generate_image,
                 inputs=[image_container, files, init_img_files, init_img_selected_idx, init_image_strength, init_image_final_weight,
                         prompt, negative_prompt, num_steps, video_length, guidance_scale, 
                         seed, attn_scale, image_embed_scale, 
                         is_adaface_enabled, adaface_ckpt_path, adaface_id_cfg_scale, adaface_power_scale, adaface_anneal_steps],
                 outputs=[result_video]
        )
    gr.Examples( fn=generate_image, examples=[], #examples, 
                 inputs=[image_container, files, init_img_files, init_img_selected_idx, init_image_strength, init_image_final_weight,
                         prompt, negative_prompt, num_steps, video_length, guidance_scale, 
                         seed, attn_scale, image_embed_scale, 
                         is_adaface_enabled, adaface_ckpt_path, adaface_id_cfg_scale, adaface_power_scale, adaface_anneal_steps], 
                 outputs=[result_video], cache_examples=True )

demo.launch(share=True, server_name=args.ip, ssl_verify=False)
