import torch
from PIL import Image
from torchvision import transforms

from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


SCI_VC_ONLY_SOURCES = {
    "VC-StrongBlur": "vcf_strong_blur",
}

SCI_VC_ONLY_EXACT_SOURCES = {
    "VC-StrongBlur-Exact": "vcf_strong_blur",
}

SCI_SINGLE_VCF_SOURCES = {
    "SCI5-StrongBlur": "vcf_strong_blur",
    "SCI5-CenterMask": "vcf_center_mask",
    "SCI5-Grayscale": "vcf_grayscale",
}
CLEAN_JOINT_SOURCES = {
    "Joint-StrongBlur-OptionShuffle1": ("vcf_strong_blur", "tcf_option_shuffle1"),
    "Joint-CenterMask-OptionShuffle1": ("vcf_center_mask", "tcf_option_shuffle1"),
    "Joint-StrongBlur-QuestionMask": ("vcf_strong_blur", "tcf_question_mask"),
    "Joint-CenterMask-QuestionMask": ("vcf_center_mask", "tcf_question_mask"),
}

SCI_VISUAL_TYPES = (
    "TIE",
    "VCD",
    "M3ID",
    "SCI3",
    "SCI5",
    "SCI7",
    "Fallback",
    "Fallback-OOD",
    "MAGIC",
    *SCI_SINGLE_VCF_SOURCES.keys(),
    *SCI_VC_ONLY_SOURCES.keys(),
    *SCI_VC_ONLY_EXACT_SOURCES.keys(),
    *CLEAN_JOINT_SOURCES.keys(),
)


def magic_option_letters(message):
    """Recover the MCQ option letters from the raw prompt text -- same
    format has_mcq_options_from_message() checks for, but returns the
    actual letters instead of a bool."""
    import re
    text = "\n".join(item.get("value", "") for item in message if item.get("type") == "text")
    return re.findall(r'(?:^|\n)([A-E])\. ', text)


def magic_letter_token_ids(tok, letters):
    """Token-id set per option letter (bare and leading-space variants),
    matching pipeline/build_mcq_bundle.py's letter_ids()."""
    out = {}
    for c in letters:
        ids = set()
        for form in (c, ' ' + c):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        out[c] = sorted(ids)
    return out


def magic_reduce_to_letters(logits_1d, letter_ids):
    """log-softmax over the full vocab, then max-over-token-variant per
    letter -- matches build_mcq_bundle.py's `max(z[t] for t in ids[c])`."""
    z = torch.log_softmax(logits_1d, dim=-1)
    return {c: max(float(z[t]) for t in ids) for c, ids in letter_ids.items() if ids}


def _magic_f32(x: float) -> float:
    """Round to float32 precision -- matches magic.py's _f32 bit-for-bit."""
    import struct
    return struct.unpack('f', struct.pack('f', x))[0]


def magic_margin(state: dict) -> float:
    """top1 - top2 over a {letter: score} dict, float32-precise -- matches
    magic.py's margin() exactly."""
    import math
    vals = sorted((_magic_f32(v) for v in state.values() if math.isfinite(v)), reverse=True)
    return _magic_f32(vals[0] - vals[1]) if len(vals) > 1 else 0.0


def magic_run_ladder(state0, lift, residual, theta1, t2, tau_v):
    """The margin ladder itself -- matches magic.py's run_ladder() exactly."""
    state = state0
    if magic_margin(state) <= theta1:
        state = lift
    if magic_margin(state) <= t2:
        cand = {c: state[c] + residual[c] for c in state}
        if magic_margin(cand) - magic_margin(state) >= tau_v:
            state = cand
    return state



# the following code is copied from Visual Contrastive Decoding
# https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_add_noise.py
def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)
    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 
    return image_tensor_cd


def has_mcq_options_from_message(message) -> bool:
    import re

    text = "\n".join(item.get("value", "") for item in message if item.get("type") == "text")
    return re.search(r"(?:^|\n)[A-D]\. .+(?:\n[A-D]\. .+)+", text) is not None


def mean_rgb(image: Image.Image) -> tuple[int, int, int]:
    small = image.convert("RGB").resize((1, 1))
    return tuple(int(v) for v in small.getpixel((0, 0)))


def center_mask_image(image: Image.Image, keep_context: float = 0.5) -> Image.Image:
    from PIL import ImageDraw

    image = image.convert("RGB")
    masked = image.copy()
    w, h = image.size
    box_w = round(w * keep_context)
    box_h = round(h * keep_context)
    left = (w - box_w) // 2
    upper = (h - box_h) // 2
    ImageDraw.Draw(masked).rectangle(
        (left, upper, left + box_w, upper + box_h),
        fill=mean_rgb(image),
    )
    return masked



def prompt_variation1(text_item):
    # new prompt text
    if '请直接回答选项字母。' in text_item:
        new_texts = text_item.replace('请直接回答选项字母。', '结合问题与选项仔细观察图像中的信息，请直接回答选项字母。')
    elif "Answer with the option's letter from the given choices directly." in text_item:
        new_texts = text_item.replace("Answer with the option's letter from the given choices directly.", "Think about the question based on details in the given image. Answer with the option's letter from the given choices directly.")
    elif 'Please answer yes or no.' in text_item:
        new_texts = text_item.replace('Please answer yes or no.', 'Think about the question based on details in the given image. Please answer yes or no.')
    elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
        new_texts = text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Think about the question based on details in the given image. Please try to answer the question with short words or phrases if possible.')
    elif 'Answer the question using a single word or phrase.' in text_item:
        new_texts = text_item.replace('Answer the question using a single word or phrase.', 'Think about the question based on details in the given image. Answer the question using a single word or phrase.')
    elif 'Answer the question directly.' in text_item:
        new_texts = text_item.replace('Answer the question directly.', 'Think about the question based on details in the given image. Answer the question directly.')
    else:
        raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation2(text_item):
    # new prompt text
    if '请直接回答选项字母。' in text_item:
        new_texts = text_item.replace('请直接回答选项字母。', 'Please carefully examine the information in the image, then consider the question and options, and reply directly with the letter corresponding to the correct answer from the options above.')
    elif "Answer with the option's letter from the given choices directly." in text_item:
        new_texts = text_item.replace("Answer with the option's letter from the given choices directly.", '请仔细观察图像中的信息，然后结合问题与选项，从上述所有选项中直接回答正确选项对应的字母。')
    elif 'Please answer yes or no.' in text_item:
        #new_texts.append(text_item.replace('Please answer yes or no.', '请直接回答yes或no。'))
        new_texts = text_item.replace('Please answer yes or no.', '观察给出的图片，请直接回答yes或no。')
    elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
        new_texts = text_item.replace('Please try to answer the question with short words or phrases if possible.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。')
    elif 'Answer the question using a single word or phrase.' in text_item:
        new_texts = text_item.replace('Answer the question using a single word or phrase.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。')
    elif 'Answer the question directly.' in text_item:
        new_texts = text_item.replace('Answer the question directly.', '请仔细观察图像中的细节，然后结合图像上的信息直接回答问题。')
    else:
        raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation3(text_item):
    # new prompt text
    if '请直接回答选项字母。' in text_item:
        new_texts = text_item.replace('请直接回答选项字母。', '你是一名擅长回答选择题的聪明学生，请直接回答选项字母。')
    elif "Answer with the option's letter from the given choices directly." in text_item:
        new_texts = text_item.replace("Answer with the option's letter from the given choices directly.", "You are a smart student who is good at answering multiple-choice questions. Answer with the option's letter from the given choices directly.")
    elif 'Please answer yes or no.' in text_item:
        new_texts = text_item.replace('Please answer yes or no.', 'You are a smart student who is good at answering yes or no questions. Please answer yes or no.')
    elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
        new_texts = text_item.replace('Please try to answer the question with short words or phrases if possible.', 'You are a smart student who is good at answering questions. Please try to answer the question with short words or phrases if possible.')
    elif 'Answer the question using a single word or phrase.' in text_item:
        new_texts = text_item.replace('Answer the question using a single word or phrase.', 'You are a smart student who is good at answering questions. Answer the question using a single word or phrase.')
    elif 'Answer the question directly.' in text_item:
        new_texts = text_item.replace('Answer the question directly.', 'You are a smart student who is good at answering questions. Answer the question directly.')
    else:
        raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_option_shuffle(text_item, seed=42):
    """TCF variant: shuffle the A/B/C/D option order to expose position bias.
    Uses content-based seed to give each sample a unique permutation, matching
    Qwen's per-sample variation (Qwen uses shared rng advancing per sample).
    """
    import re
    import random
    import hashlib
    content_hash = int(hashlib.md5(text_item.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed ^ content_hash)
    option_pat = re.compile(r'((?:[A-D]\. .+\n?)+)', re.MULTILINE)
    match = option_pat.search(text_item)
    if match is None:
        return text_item
    block = match.group(0)
    line_pat = re.compile(r'^([A-D])\. (.+)$', re.MULTILINE)
    options = line_pat.findall(block)
    if len(options) < 2:
        return text_item
    labels = [o[0] for o in options]
    contents = [o[1] for o in options]
    perm = list(range(len(contents)))
    rng.shuffle(perm)
    shuffled_block = '\n'.join(
        f'{labels[i]}. {contents[perm[i]]}' for i in range(len(labels))
    ) + '\n'
    return text_item[:match.start()] + shuffled_block + text_item[match.end():]


def prompt_paraphrase1(text_item):
    if '请直接回答选项字母。' in text_item:
        new_texts = text_item.replace('请直接回答选项字母。', '请根据图像内容判断正确选项，并且只输出对应的选项字母。')
    elif "Answer with the option's letter from the given choices directly." in text_item:
        new_texts = text_item.replace("Answer with the option's letter from the given choices directly.", "Use the image to choose the correct option. Respond with only the option letter.")
    elif 'Please answer yes or no.' in text_item:
        new_texts = text_item.replace('Please answer yes or no.', 'Use the image to decide the answer. Respond only with yes or no.')
    elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
        new_texts = text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Use the image to answer concisely, preferably with a short word or phrase.')
    elif 'Answer the question using a single word or phrase.' in text_item:
        new_texts = text_item.replace('Answer the question using a single word or phrase.', 'Use the image to answer directly with one word or a short phrase.')
    elif 'Answer the question directly.' in text_item:
        new_texts = text_item.replace('Answer the question directly.', 'Use the image to answer directly.')
    else:
        raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_answer_format1(text_item):
    if '请直接回答选项字母。' in text_item:
        new_texts = text_item.replace('请直接回答选项字母。', '请只输出最终答案的选项字母，不要输出解释。')
    elif "Answer with the option's letter from the given choices directly." in text_item:
        new_texts = text_item.replace("Answer with the option's letter from the given choices directly.", "Return only the final option letter, with no explanation.")
    elif 'Please answer yes or no.' in text_item:
        new_texts = text_item.replace('Please answer yes or no.', 'Return exactly one word: yes or no.')
    elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
        new_texts = text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Return only the final short answer, without a full sentence.')
    elif 'Answer the question using a single word or phrase.' in text_item:
        new_texts = text_item.replace('Answer the question using a single word or phrase.', 'Return only the final short answer, without a full sentence.')
    elif 'Answer the question directly.' in text_item:
        new_texts = text_item.replace('Answer the question directly.', 'Return only the final answer, without a full sentence.')
    else:
        raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


# --- TCF grounding probes (ported from Qwen2-VL, single-string LLaVA convention) ---
def _prompt_body_start(text_item, limit):
    markers = ['<|vision_end|>', '<image>']
    best = -1
    best_marker = ''
    for marker in markers:
        pos = text_item.rfind(marker, 0, limit)
        if pos > best:
            best = pos
            best_marker = marker
    if best >= 0:
        return best + len(best_marker)
    return 0


def _instruction_start(text_item):
    markers = [
        '请直接回答选项字母。',
        'Please select the correct answer from the options above.',
        'Please answer yes or no.',
        'Please try to answer the question with short words or phrases if possible.',
        'Answer the question directly using a single word or phrase.',
        'Answer the question directly.',
    ]
    positions = [text_item.find(marker) for marker in markers]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return -1
    return min(positions)


def _mask_question_words(segment):
    import re

    stopwords = {
        'about', 'above', 'after', 'also', 'answer', 'based', 'before',
        'between', 'choose', 'correct', 'directly', 'does', 'following',
        'from', 'given', 'image', 'option', 'options', 'please', 'question',
        'select', 'shown', 'that', 'there', 'these', 'this', 'using', 'what',
        'when', 'where', 'which', 'will', 'with',
    }

    if re.search(r'[一-鿿]', segment):
        stripped = segment.strip()
        if not stripped:
            return segment
        leading = segment[:len(segment) - len(segment.lstrip())]
        trailing = segment[len(segment.rstrip()):]
        return f'{leading}The key question words are intentionally masked.{trailing}'

    def replace_word(match):
        word = match.group(0)
        lower = word.lower()
        if len(word) < 4 or lower in stopwords:
            return word
        return 'something'

    return re.sub(r'\b[A-Za-z][A-Za-z0-9-]{3,}\b', replace_word, segment)


def prompt_options_only(text_item):
    """TCF prior probe: remove the real question, keep answer options."""
    import re
    option_pat = re.compile(r'((?:[A-D]\. .+\n?)+)', re.MULTILINE)
    match = option_pat.search(text_item)
    if match is not None:
        body_start = _prompt_body_start(text_item, match.start())
        replacement = '\nQuestion content is intentionally hidden. Choose the most plausible answer from the options only.\n'
        return text_item[:body_start] + replacement + text_item[match.start():]
    instr_start = _instruction_start(text_item)
    if instr_start >= 0:
        body_start = _prompt_body_start(text_item, instr_start)
        replacement = '\nQuestion content is intentionally hidden. Use only the answer format prior. '
        return text_item[:body_start] + replacement + text_item[instr_start:]
    return text_item


def prompt_question_mask(text_item):
    """TCF grounding probe: mask question content while preserving format."""
    import re
    option_pat = re.compile(r'((?:[A-D]\. .+\n?)+)', re.MULTILINE)
    match = option_pat.search(text_item)
    if match is not None:
        body_start = _prompt_body_start(text_item, match.start())
        segment = text_item[body_start:match.start()]
        masked = _mask_question_words(segment)
        if masked == segment:
            masked = '\nThe key question words are intentionally masked.\n'
        return text_item[:body_start] + masked + text_item[match.start():]
    instr_start = _instruction_start(text_item)
    if instr_start >= 0:
        body_start = _prompt_body_start(text_item, instr_start)
        segment = text_item[body_start:instr_start]
        masked = _mask_question_words(segment)
        if masked == segment:
            masked = '\nThe key question words are intentionally masked. '
        return text_item[:body_start] + masked + text_item[instr_start:]
    return text_item


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="liuhaotian/llava_v1.5_7b", **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as err:
            logging.critical(
                "Please install llava from https://github.com/haotian-liu/LLaVA"
            )
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        if model_path == "Lin-Chen/ShareGPT4V-7B":
            model_name = "llava-v1.5-7b"
        elif model_path == "Lin-Chen/ShareGPT4V-13B":
            model_name = "llava-v1.5-13b"
        else:
            model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="cpu",
                )
            )
        except Exception as err:
            if "ShareGPT4V" in model_path:
                import llava

                logging.critical(
                    "Please manually remove the encoder type check in "
                    f"{llava.__path__[0]}/model/multimodal_encoder/builder.py "
                    "Line 8 to use the ShareGPT4V model. "
                )
            else:
                logging.critical("Unknown error when loading LLaVA model.")
            raise err

        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            content, images_sub = self.concat_tilist(utter["content"])
            prompt += content
            images.extend(images_sub)
            prompt += " " if utter["role"] == "user" else self.stop_str
        assert message[-1]["role"] == "user", message
        prompt += "ASSISTANT: "

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, self.image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self, 
        model_path="llava-hf/llava-v1.6-vicuna-7b-hf", 
        model_name=None,
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type='default',
        textual_type='default',
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        theta=0.0,
        magic_theta1=0.0,
        magic_t2=0.0,
        magic_tau_v=0.0,
        **kwargs):
        import transformers
        from transformers import (
            LlavaNextProcessor,
            AutoProcessor,
            LlavaForConditionalGeneration,
        )

        # custom parameters
        self.visual_type = visual_type
        self.textual_type = textual_type
        print(f"==> Using Image type: {visual_type}")
        print(f"==> Using Prompt type: {textual_type}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.magic_theta1 = magic_theta1
        self.magic_t2 = magic_t2
        self.magic_tau_v = magic_tau_v

        self.model_path = model_path
        if "34b" in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_path, use_fast=False
            )
        elif "interleave" in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        flash_attn_flag = False
        try:
            import flash_attn

            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
            else:
                from .modeling_llava_next import LlavaNextForConditionalGeneration
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
        else:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
                )
            else:
                from .modeling_llava_next import LlavaNextForConditionalGeneration
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
                )

        model = model.eval()
        self.model = model.cuda()
        if self.visual_type in SCI_VISUAL_TYPES:
            kwargs_default = dict(
                do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1,
                return_dict_in_generate=True,   # <‑‑ ask for a dict‑like GenerationOutput
                output_logits=True,
            )
        else:
            kwargs_default = dict(
                do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1
            )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

        # Kaihua Modified
        if save_logits and (model_name is not None) and (dump_path is not None):
            save_path = os.path.join(dump_path, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Saving logit tensors to {save_path}")
            self.model.dump_path = save_path
        else:
            self.model.dump_path = None

    def apply_prompt_template(self, prompt):
        model_path = self.model_path.lower()
        if "mistral" in model_path:
            template = "[INST] PLACEHOLDER [/INST]"
        elif "vicuna" in model_path:
            template = (
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                "USER: PLACEHOLDER ASSISTANT:"
            )
        elif "34b" in model_path:
            template = (
                "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>"
                "<|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError(
                f"Prompt template for {model_path} not implemented."
            )

        prompt = template.replace("PLACEHOLDER", f"<image>\n{prompt}")
        return prompt

    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message
    
    def process_message(self, message):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return prompt, images

    def generate_ids_or_logits(self, messages, visual_type, textual_type, cf_logits=None, cf_params=None, get_logits=False):
        prompt, images = self.process_message(message=messages)
        
        # process image
        processed_images = []
        if visual_type in ('default', *SCI_VISUAL_TYPES):
            processed_images = images
        elif visual_type == 'vcf_color0':
            for image in images:
                black_image = Image.new("RGB", image.size, (0, 0, 0))
                processed_images.append(black_image)
        elif visual_type == 'vcf_color255':
            for image in images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                processed_images.append(white_image)
        elif visual_type == 'vcf_noise400':
            for image in images:
                noise_img = add_diffusion_noise(transforms.ToTensor()(image), noise_step=400)
                processed_images.append(transforms.ToPILImage()(noise_img.cpu()))
        elif visual_type == 'vcf_noise500':
            for image in images:
                noise_img = add_diffusion_noise(transforms.ToTensor()(image), noise_step=500)
                processed_images.append(transforms.ToPILImage()(noise_img.cpu()))
        elif visual_type == 'vcf_strong_blur':
            from PIL import ImageFilter
            for image in images:
                processed_images.append(image.filter(ImageFilter.GaussianBlur(radius=100)))
        elif visual_type == 'vcf_center_mask':
            for image in images:
                processed_images.append(center_mask_image(image))
        elif visual_type == 'vcf_grayscale':
            for image in images:
                processed_images.append(image.convert('L').convert('RGB'))
        elif visual_type == 'vcf_high_pass':
            # High-pass — remove LOW frequency, keep edges/high freq (frequency-complement of blur).
            from PIL import ImageFilter, ImageChops
            for image in images:
                low = image.filter(ImageFilter.GaussianBlur(radius=50))
                processed_images.append(ImageChops.subtract(image, low, scale=1.0, offset=128))
        else:
            raise ValueError("Wrong Image Type")

        # process text
        if textual_type in ('default', *SCI_VISUAL_TYPES):
            processed_texts = prompt
        elif textual_type == 'tcf_v1':
            processed_texts = prompt_variation1(prompt)
        elif textual_type == 'tcf_v2':
            processed_texts = prompt_variation2(prompt)
        elif textual_type == 'tcf_v3':
            processed_texts = prompt_variation3(prompt)
        elif textual_type == 'tcf_option_shuffle1':
            processed_texts = prompt_option_shuffle(prompt, seed=42)
        elif textual_type == 'tcf_option_shuffle2':
            processed_texts = prompt_option_shuffle(prompt, seed=137)
        elif textual_type == 'tcf_paraphrase1':
            processed_texts = prompt_paraphrase1(prompt)
        elif textual_type == 'tcf_answer_format1':
            processed_texts = prompt_answer_format1(prompt)
        elif textual_type == 'tcf_options_only':
            processed_texts = prompt_options_only(prompt)
        elif textual_type == 'tcf_question_mask':
            processed_texts = prompt_question_mask(prompt)
        else:
            raise ValueError("Wrong Text Type")

        # inference
        if visual_type in SCI_VISUAL_TYPES:
            assert (cf_logits is not None) and (cf_params is not None)
            inputs = self.processor(processed_texts, processed_images, return_tensors="pt").to("cuda", torch.float16)
            output_dicts = self.model.generate(
                    cf_logits = cf_logits,
                    cf_params = cf_params,
                    **inputs,
                    **self.kwargs,
                )
            logits = torch.cat(output_dicts.logits, dim=0)
            output = logits.unsqueeze(0).argmax(dim=-1)
            return output
        else:
            inputs = self.processor(processed_texts, processed_images, return_tensors="pt").to("cuda", torch.float16)
            if get_logits:
                output_dicts = self.model.generate(
                    **inputs,
                    **self.kwargs,
                )
                logits = torch.cat(output_dicts.logits, dim=0)
                return logits.unsqueeze(0).float()
            else:
                output = self.model.generate(**inputs, **self.kwargs)
                return output

    def generate_inner(self, message, dataset=None):
        if self.visual_type == 'TIE':
            # We inplement TIE based on paper Counterfactual VQA and Unbiased Scene Graph Generation
            tie_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"tie_logits": tie_logits}
            cf_params = {"theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'VCD':
            # Visual Contrastive Decoding (VCD)
            # Current BS-Subsets only predict one character or one word, so we don't need to take care of auto-regressive generation, focusing on the first token is enough. 
            # It can be generalized to iterative generation in future work if needed.
            vcd_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"vcd_logits": vcd_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'M3ID':
            m3id_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"m3id_logits": m3id_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI3':
            vcf1_logits = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v1', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "tcf1_logits": tcf1_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI5':
            # The proposed self-critical inference
            vcf1_logits = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in SCI_SINGLE_VCF_SOURCES:
            vcf_logits = self.generate_ids_or_logits(
                message,
                visual_type=SCI_SINGLE_VCF_SOURCES[self.visual_type],
                textual_type='default',
                get_logits=True,
            )
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"vcf1_logits": vcf_logits, "vcf2_logits": vcf_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Fallback':
            vcf1_logits = self.generate_ids_or_logits(message, visual_type='vcf_strong_blur', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(message, visual_type='vcf_center_mask', textual_type='default', get_logits=True)
            if has_mcq_options_from_message(message):
                tcf1_type, tcf2_type = 'tcf_option_shuffle1', 'tcf_option_shuffle2'
            else:
                tcf1_type, tcf2_type = 'tcf_question_mask', 'tcf_options_only'
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type=tcf1_type, get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type=tcf2_type, get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Fallback-OOD':
            # Option-2 floor: clean format-aware TC probes (same as Fallback) but VC = FULL
            # ablation (black + noise) instead of partial blur/mask.
            vcf1_logits = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            if has_mcq_options_from_message(message):
                tcf1_type, tcf2_type = 'tcf_option_shuffle1', 'tcf_option_shuffle2'
            else:
                tcf1_type, tcf2_type = 'tcf_question_mask', 'tcf_options_only'
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type=tcf1_type, get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type=tcf2_type, get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in CLEAN_JOINT_SOURCES:
            visual_source, text_source = CLEAN_JOINT_SOURCES[self.visual_type]
            vcf_logits = self.generate_ids_or_logits(message, visual_type=visual_source, textual_type='default', get_logits=True)
            tcf_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type=text_source, get_logits=True)
            cf_logits = {"vcf1_logits": vcf_logits, "tcf1_logits": tcf_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in SCI_VC_ONLY_SOURCES:
            orig_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='default', get_logits=True)
            vcf1_logits = self.generate_ids_or_logits(
                message,
                visual_type=SCI_VC_ONLY_SOURCES[self.visual_type],
                textual_type='default',
                get_logits=True,
            )
            cf_logits = {"vcf1_logits": vcf1_logits, "tcf1_logits": orig_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in SCI_VC_ONLY_EXACT_SOURCES:
            vcf_logits = self.generate_ids_or_logits(
                message,
                visual_type=SCI_VC_ONLY_EXACT_SOURCES[self.visual_type],
                textual_type='default',
                get_logits=True,
            )
            cf_logits = {"vc_only_logits": vcf_logits}
            cf_params = {"beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI7':
            vcf1_logits = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            vcf3_logits = self.generate_ids_or_logits(message, visual_type='vcf_noise400', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v2', get_logits=True)
            tcf3_logits = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_v3', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "vcf3_logits": vcf3_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits, "tcf3_logits": tcf3_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'MAGIC':
            # Live MAGIC: same margin ladder magic.py replays offline, one
            # call, internally multi-pass. MCQ only -- see the qwen2_vl
            # MAGIC branch for the full rationale; this mirrors it, adjusted
            # for LLaVA's own generation convention below.
            #
            # NOTE index [:,1,:] not [:,0,:]: LLaVA-NeXT's first generated
            # token is a newline on this backbone, the answer is the second
            # (this is the same position issue the offline Oth pipeline hit
            # -- see pipeline/README.md's disclosed history) -- confirmed by
            # every existing cf_logits consumer in modeling_llava_next.py
            # indexing [:,1,:], never [:,0,:].
            letters = magic_option_letters(message)
            if not letters:
                raise ValueError(
                    "visual_type='MAGIC' is an MCQ-only live path (needs "
                    "'A. ...' / 'B. ...' options in the prompt); open-ended "
                    "MAGIC scoring is not implemented at generation time -- "
                    "see pipeline/build_oth_bundle.py for the offline path."
                )
            tok = self.processor.tokenizer
            letter_ids = magic_letter_token_ids(tok, letters)

            real_def = self.generate_ids_or_logits(message, visual_type='default', textual_type='default', get_logits=True)[:, 1, :].float()
            blank_def = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='default', get_logits=True)[:, 1, :].float()
            real_ans = self.generate_ids_or_logits(message, visual_type='default', textual_type='tcf_answer_format1', get_logits=True)[:, 1, :].float()
            blank_ans = self.generate_ids_or_logits(message, visual_type='vcf_color0', textual_type='tcf_answer_format1', get_logits=True)[:, 1, :].float()
            gray = self.generate_ids_or_logits(message, visual_type='vcf_grayscale', textual_type='default', get_logits=True)[:, 1, :].float()
            strongblur = self.generate_ids_or_logits(message, visual_type='vcf_strong_blur', textual_type='default', get_logits=True)[:, 1, :].float()
            centermask = self.generate_ids_or_logits(message, visual_type='vcf_center_mask', textual_type='default', get_logits=True)[:, 1, :].float()

            s0 = magic_reduce_to_letters(real_def[0], letter_ids)
            bd = magic_reduce_to_letters(blank_def[0], letter_ids)
            ra = magic_reduce_to_letters(real_ans[0], letter_ids)
            ba = magic_reduce_to_letters(blank_ans[0], letter_ids)
            default_delta = {c: s0[c] - bd[c] for c in s0}
            answer_delta = {c: ra[c] - ba[c] for c in s0}
            lift = {c: max(default_delta[c], answer_delta[c]) for c in s0}

            orig_conf = torch.softmax(real_def[0], dim=-1).max().item()
            channels = {'gray': gray, 'strongblur': strongblur, 'centermask': centermask}
            iic = {name: orig_conf - torch.softmax(v[0], dim=-1).max().item() for name, v in channels.items()}
            routed = min(iic, key=iic.get)
            vc = magic_reduce_to_letters(channels[routed][0], letter_ids)
            residual = {c: s0[c] - vc[c] for c in s0}

            state = magic_run_ladder(s0, lift, residual, self.magic_theta1, self.magic_t2, self.magic_tau_v)
            pred_letter = max(state, key=state.get)
            output = torch.tensor([letter_ids[pred_letter][:1]], device=real_def.device)
        else:
            output = self.generate_ids_or_logits(message, visual_type=self.visual_type, textual_type=self.textual_type)

        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        return answer


class LLaVA_Next2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llama3-llava-next-8b", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=None
        )
        model.cuda().eval()
        model.tie_weights()

        if "llama3" in model_path.lower():
            conv_mode = "llava_llama_3"
        elif "qwen" in model_path.lower():
            conv_mode = "qwen_1_5"
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors="pt")["pixel_values"][0].half().cuda()
            for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    # This function is used to split InternVL2-Llama3-76B
    def split_model(self, model_path):
        import math

        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size
        if "72b" not in model_path.lower():
            return None
        # embed_tokens, vision_tower, mm_projector, lm_head are treated as 2 layers
        num_layers = 80 + 8
        num_layers_per_gpu = math.ceil(num_layers / num_gpus)
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 6
        num_layers_per_gpu[-1] -= 2
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"model.layers.{layer_cnt}"] = rank + world_size * i
                layer_cnt += 1
        last_gpu = rank + world_size * (num_gpus - 1)
        device_map["model.image_newline"] = rank
        device_map["model.embed_tokens"] = rank
        device_map["model.norm"] = rank
        device_map["model.vision_tower"] = rank
        device_map["model.vision_resampler"] = rank
        device_map["model.mm_projector"] = rank
        device_map["lm_head"] = last_gpu
        return device_map

    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )  # noqa: E501
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        video_kwargs_default = dict(
            overwrite=True, mm_spatial_pool_mode="average", force_sample=True
        )
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs[
                    "mm_spatial_pool_mode"
                ]

        rank, world_size = get_rank_and_world_size()
        model_name = get_model_name_from_path(model_path)
        device_map = self.split_model(model_path)

        if device_map is None:
            if auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT set for non-72B LLaVA-OneVision'
                logging.warning('Currently, we only support to split the non-72B model across all GPUs.')
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    None,
                    model_name,
                    device_map="auto",
                    overwrite_config=overwrite_config,
                )
            else:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    None,
                    model_name,
                    device_map="cpu",
                    overwrite_config=overwrite_config,
                )
                model.cuda()
        else:
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map=device_map,
                overwrite_config=overwrite_config,
            )
        model.eval()
        model.tie_weights()

        if "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(
            images, self.image_processor, self.model.config
        )
        image_tensor = [
            _image.to(dtype=torch.float16, device="cuda") for _image in image_tensor
        ]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError(
                "LLaVA-OneVision does not support multiple videos as input."
            )

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, 1, self.force_sample
        )

        time_instruciton = (
            f"The video lasts for {video_time:.2f} seconds,"
            f"and {len(video_frames[0])} frames are uniformly sampled from it."
            f"These frames are located at {frame_time}."
            f"Please answer the following questions related to this video.\n"
        )

        if self.force_sample:
            content = visual_content + time_instruciton + text_content
        else:
            content = visual_content + text_content

        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def load_video(self, video_path, max_frames_num, force_sample=False, fps=1):
        from decord import VideoReader, cpu
        import numpy as np

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, sample_fps, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames, frame_time, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)


class LLaVA_OneVision_HF(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1
        self.model_path = model_path

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError("LLaVA-OneVision does not support multiple videos as input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, fps=1, force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": content}, {"type": "video"}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_fps = round(avg_fps / fps)
        frame_idx = list(range(0, total_frame_num, effective_fps))
        frame_time = [i / avg_fps for i in frame_idx]

        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])
        video_frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frame_num / avg_fps

        return video_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO":
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)
