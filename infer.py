from diffusers import AutoencoderKL, DDIMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from omegaconf import OmegaConf
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import load_weights
from safetensors import safe_open
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from faceadapter.face_adapter import FaceAdapterPlusForVideoLora
from adaface.adaface_wrapper import AdaFaceWrapper

def load_adaface(base_model_path, adaface_ckpt_path, device="cuda"):
    # base_model_path is only used for initialization, not really used in the inference.
    adaface = AdaFaceWrapper(pipeline_name="text2img", base_model_path=base_model_path,
                             adaface_ckpt_path=adaface_ckpt_path, device=device)
    return adaface

def load_model(base_model_type="rv40", adaface_base_model_type="sd15", 
               adaface_ckpt_path=None, device="cuda:0"):
    inference_config    = "inference-v2.yaml"
    sd_version          = "animatediff/sd"
    id_ckpt             = "models/animator.ckpt"
    image_encoder_path  = "image_encoder"

    base_model_type_to_path = {
        "rv40": "models/realisticvision/realisticVisionV40_v40VAE.safetensors",
        "rv60": "models/realisticvision/realisticVisionV60B1_v51VAE.safetensors",
        "sd15": "models/stable-diffusion-v-1-5/v1-5-pruned.safetensors",
        "sd15_adaface": "models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt",
        "toonyou": "models/toonyou/toonyou_beta6.safetensors",
        "epv5": "models/epic_realism/epicrealism_pureEvolutionV5.safetensors",
        "ar181": "models/absolutereality/absolutereality_v181.safetensors",
        "ar16":  "models/absolutereality/ar-v1-6.safetensors",
        "sar":   "models/sar/sar.safetensors",
    }

    base_model_path         = base_model_type_to_path[base_model_type]
    if adaface_base_model_type + "_adaface" in base_model_type_to_path:
        adaface_base_model_path = base_model_type_to_path[adaface_base_model_type + "_adaface"]
    else:
        adaface_base_model_path = base_model_type_to_path[adaface_base_model_type]

    motion_module_path="models/v3_sd15_mm.ckpt" 
    motion_lora_path = "models/v3_sd15_adapter.ckpt"
    inference_config = OmegaConf.load(inference_config)    

    tokenizer    = CLIPTokenizer.from_pretrained(sd_version, subfolder="tokenizer",torch_dtype=torch.float16,
    )
    text_encoder = CLIPTextModel.from_pretrained(sd_version, subfolder="text_encoder",torch_dtype=torch.float16,
    ).to(device=device)
    vae          = AutoencoderKL.from_pretrained(sd_version, subfolder="vae",torch_dtype=torch.float16,
    ).to(device=device)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_version, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
    ).to(device=device)
    pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=None,
            #beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=DPMSolverMultistepScheduler(**OmegaConf.to_container(inference_config.DPMSolver_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1

    ),torch_dtype=torch.float16,
            ).to(device=device)
    
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
    ).to(device=device)
    if base_model_path != "":
        print(f"load dreambooth model from {base_model_path}")
        dreambooth_state_dict = {}
        with safe_open(base_model_path, framework="pt", device="cpu") as f:
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

            pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict).to(device=device)
            
        del dreambooth_state_dict
        pipeline = pipeline.to(torch.float16)
        id_animator = FaceAdapterPlusForVideoLora(pipeline, image_encoder_path, id_ckpt, num_tokens=16,
                                                  device=torch.device(device), torch_type=torch.float16)

        if adaface_ckpt_path is not None:
            adaface = load_adaface(adaface_base_model_path, #dreambooth_model_path, 
                                   adaface_ckpt_path, device)
        else:
            adaface = None

        return id_animator, adaface
    
