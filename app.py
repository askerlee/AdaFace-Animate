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
import numpy as np
import argparse
# From command line read command embman_ckpt_path
parser = argparse.ArgumentParser()
parser.add_argument('--embman_ckpt_path', type=str, 
                    default='/data/shaohua/adaprompt/logs/subjects-celebrity2024-05-16T17-22-46_zero3-ada/checkpoints/embeddings_gs-30000.pt')
args = parser.parse_args()

# model = load_model()
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

id_animator, adaface = load_model(embman_ckpt_path=args.embman_ckpt_path)
basedir     = os.getcwd()
savedir     = os.path.join(basedir,'samples')
os.makedirs(savedir, exist_ok=True)

#print(f"### Cleaning cached examples ...")
#os.system(f"rm -rf gradio_cached_examples/")

@spaces.GPU
def generate_image(image_container, uploaded_image_paths, prompt, negative_prompt, 
                   num_steps, video_length, guidance_scale, seed, attn_scale, image_embed_scale,
                   is_adaface_enabled, embman_ckpt_path, adaface_scale, adaface_anneal_steps,
                   progress=gr.Progress(track_tqdm=True)):
    # check the trigger word
    # apply the style template

    prompt = prompt + " 8k uhd, high quality"
    if " shot" not in prompt:
        prompt = prompt + ", medium shot"
        
    prompt_img_lists=[]
    for path in uploaded_image_paths:
        img = cv2.imread(path)
        faces = app.get(img)
        face_roi = face_align.norm_crop(img,faces[0]['kps'],112)
        random_name = str(uuid.uuid4())
        face_path = os.path.join(savedir, f"{random_name}.jpg")
        cv2.imwrite(face_path, face_roi)
        # prompt_img_lists is a list of PIL images.
        prompt_img_lists.append(load_image(face_path).resize((224,224)))

    if adaface is None or not is_adaface_enabled:
        adaface_prompt_embeds = None
    else:
        if embman_ckpt_path != args.embman_ckpt_path:
            # Reload the embedding manager
            adaface.load_subj_basis_generator(embman_ckpt_path)

        adaface.generate_adaface_embeddings(image_folder=None, image_paths=uploaded_image_paths,
                                            out_id_embs_scale=adaface_scale,
                                            update_text_encoder=True)
        # adaface_prompt_embeds: [1, 77, 768].
        adaface_prompt_embeds, _ = adaface.encode_prompt(prompt)

    sample = id_animator.generate(prompt_img_lists, 
                                  init_image      = None, #init_image,
                                  prompt = prompt,
                                  negative_prompt = negative_prompt + " long shots, full body",
                                  adaface_embeds  = adaface_prompt_embeds,
                                  # This adaface_scale is not so useful, and when it's set >= 2, weird artifacts appear. So we disable it,
                                  # and apply adaface_scale in the generate_adaface_embeddings() function.
                                  adaface_scale   = 1,
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

def validate(prompt):
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
        # ID-Animator: Zero-Shot Identity-Preserving Human Video Generation
        Xuanhua He, Quande Liu✉, Shengju Qian,Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Jie Zhang✉ (✉Corresponding Author)<br>
        [Arxiv Report](https://arxiv.org/abs/2404.15275) | [Project Page](https://id-animator.github.io/) | [Github](https://github.com/ID-Animator/ID-Animator)
        """
    )
    gr.Markdown(
        """
    ❗️❗️❗️**Tips:**
    - we provide some examples in the bottom, you can try these example prompts first
    - you can upload one image for generating ID-specific video or upload multiple images for mixing different IDs
    - Adjust the image scale to enhance the generation quality. A larger image scale improves the ability to preserve identity, while a smaller image scale enhances the ability to follow instructions.
        """
    )

    with gr.Row():
        with gr.Column():
            files = gr.File(
                        label="Drag (Select) 1 or more photos of your face",
                        file_types=["image"],
                        file_count="multiple"
                    )
            image_container = gr.Image(label="image container", sources="upload", type="numpy", height=256,visible=False)
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                    #    info="Try something like 'a photo of a man/woman img', 'img' is the trigger word.",
                       placeholder="Iron Man soars through the clouds, his repulsors blazing.")
           
            attn_scale = gr.Slider(
                    label="Attention Processor Scale",
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=1,
                )
            image_embed_scale = gr.Slider(
                    label="Image Embedding Scale",
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=0.5,
                )
            adaface_scale = gr.Slider(
                    label="AdaFace Embedding Scale",
                    minimum=0,
                    maximum=4,
                    step=0.1,
                    value=1,
                )
             
            submit = gr.Button("Submit")

            with gr.Accordion(open=False, label="Advanced Options"):
                video_length = gr.Slider(
                    label="video_length",
                    minimum=16,
                    maximum=21,
                    step=1,
                    value=16,
                )
                is_adaface_enabled = gr.Checkbox(label="Enable AdaFace", value=True)
                adaface_anneal_steps = gr.Slider(
                    label="AdaFace Anneal Steps",
                    minimum=0,
                    maximum=2,
                    step=1,
                    value=0,
                )
                embman_ckpt_path = gr.Textbox(
                    label="AdaFace ckpt Path", 
                    placeholder=args.embman_ckpt_path,
                    value=args.embman_ckpt_path,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, bare breasts, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, long neck, UnrealisticDream",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=25,
                    maximum=100,
                    step=1,
                    value=40,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
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
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
        with gr.Column():
            result_video = gr.Video(label="Generated Animation", interactive=False)
        
        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])
        submit.click(fn=validate,
                     inputs=[prompt],outputs=None).success(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
                 fn=generate_image,
                 inputs=[image_container, files, prompt, negative_prompt, num_steps, video_length, guidance_scale, 
                         seed, attn_scale, image_embed_scale, is_adaface_enabled, embman_ckpt_path, adaface_scale, adaface_anneal_steps],
                 outputs=[result_video]
        )
    gr.Examples( fn=generate_image, examples=[], #examples, 
                 inputs=[image_container, files, prompt, negative_prompt, num_steps, video_length, guidance_scale, 
                         seed, attn_scale, image_embed_scale, is_adaface_enabled, embman_ckpt_path, adaface_scale, adaface_anneal_steps], 
                 outputs=[result_video], cache_examples=True )

demo.launch(share=True)
