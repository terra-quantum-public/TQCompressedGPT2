import utils
from transformers import AutoModelForCausalLM


def TQCompressedGPT2ForCausalLM():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model_ = utils.parser.get_model_cfg(
        "config/config.yaml", model, compute=False, q_num=-1
    )
    return model_
