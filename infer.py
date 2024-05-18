from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from omegaconf import OmegaConf
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import load_weights
from safetensors import safe_open
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from faceadapter.face_adapter import FaceAdapterPlusForVideoLora
from diffusers.utils import load_image
from animatediff.utils.util import save_videos_grid
from huggingface_hub import hf_hub_download
from ldm.util import instantiate_from_config

def load_ddpm(unet_ckpt_path, embman_ckpt_path):
    config = OmegaConf.load("/home/shaohua/adaprompt/configs/stable-diffusion/v1-inference-ada-empty.yaml")
    config.model.params.personalization_config.params.token2num_vectors = { 'z': 16, 'y': 4 }
    config.model.params.personalization_config.params.skip_loading_token2num_vectors = True

    # We don't instantiate and load UNet weights, as they are not used.
    # Doing so will reduce RAM usage greatly.
    ddpm = instantiate_from_config(config.model)
    ddpm.embedding_manager.load(embman_ckpt_path, load_old_embman_ckpt=False)
    ddpm.embedding_manager.eval()

    # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    ddpm.cond_stage_model.set_last_layers_skip_weights([0.5, 0.5])    
    ddpm  = ddpm.to("cuda")
    ddpm.cond_stage_model.device = "cuda"

    return ddpm

def load_model(embman_ckpt_path=None):
    inference_config = "inference-v2.yaml"
    sd_version = "animatediff/sd"
    id_ckpt = "animator.ckpt"
    image_encoder_path = "image_encoder"
    dreambooth_model_path = "realisticVisionV60B1_v51VAE.safetensors"
    motion_module_path="v3_sd15_mm.ckpt" #"mm_sd_v15_v2.ckpt"
    motion_lora_path = "v3_sd15_adapter.ckpt"
    inference_config = OmegaConf.load(inference_config)    


    tokenizer    = CLIPTokenizer.from_pretrained(sd_version, subfolder="tokenizer",torch_dtype=torch.float16,
    )
    text_encoder = CLIPTextModel.from_pretrained(sd_version, subfolder="text_encoder",torch_dtype=torch.float16,
    ).cuda()
    vae          = AutoencoderKL.from_pretrained(sd_version, subfolder="vae",torch_dtype=torch.float16,
    ).cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(sd_version, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
    ).cuda()
    pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=None,
            #beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=DPMSolverMultistepScheduler(**OmegaConf.to_container(inference_config.DPMSolver_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1

    ),torch_dtype=torch.float16,
            ).to("cuda")
    
    pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = motion_module_path,
            motion_module_lora_configs = [],
            # domain adapter
            adapter_lora_path          = motion_lora_path,
            adapter_lora_scale         = 1,
            # image layers
            dreambooth_model_path      = None,
            lora_model_path            = "",
            lora_alpha                 = 0.8
    ).to("cuda")
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        dreambooth_state_dict = {}
        with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
                        
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
            # print(vae)
            #vae ->to_q,to_k,to_v
            # print(converted_vae_checkpoint)
            convert_vae_keys = list(converted_vae_checkpoint.keys())
            for key in convert_vae_keys:
                if "encoder.mid_block.attentions" in key or "decoder.mid_block.attentions" in  key:
                    new_key = None
                    if "key" in key:
                        new_key = key.replace("key","to_k")
                    elif "query" in key:
                        new_key = key.replace("query","to_q")
                    elif "value" in key:
                        new_key = key.replace("value","to_v")
                    elif "proj_attn" in key:
                        new_key = key.replace("proj_attn","to_out.0")
                    if new_key:
                        converted_vae_checkpoint[new_key] = converted_vae_checkpoint.pop(key)

            pipeline.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
            pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict).to("cuda")
        del dreambooth_state_dict
        pipeline = pipeline.to(torch.float16)
        id_animator = FaceAdapterPlusForVideoLora(pipeline, image_encoder_path, id_ckpt, num_tokens=16,device=torch.device("cuda"),torch_type=torch.float16)

        if embman_ckpt_path is not None:
            ddpm = load_ddpm(dreambooth_model_path, embman_ckpt_path)
        else:
            ddpm = None

        return id_animator, ddpm
    
