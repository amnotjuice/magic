from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr

try:
    from qwen_vl_utils import process_vision_info
except Exception as err:
    logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
    raise err

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
    'TIE', 'VCD', 'M3ID', 'SCI3', 'SCI5', 'SCI7',
    'Ablation1', 'Ablation2', 'SCI-Adaptive', 'Fallback', 'Fallback-OOD',
    'MAGIC',
    *SCI_SINGLE_VCF_SOURCES.keys(),
    *CLEAN_JOINT_SOURCES.keys(),
)


def magic_option_letters(message):
    """Recover the MCQ option letters (e.g. ['A','B','C','D']) from the raw
    prompt text in `message`, the same "A. ...\\nB. ..." block format used
    by prompt_option_shuffle. Returns [] if this isn't an MCQ prompt."""
    import re
    text = ' '.join(m.get('value', '') for m in message if m.get('type') == 'text')
    return re.findall(r'^([A-E])\. ', text, re.MULTILINE)


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
    magic.py's margin() exactly so the live ladder makes the same allocation
    and commitment decisions the offline replay does."""
    vals = sorted((_magic_f32(v) for v in state.values() if math.isfinite(v)), reverse=True)
    return _magic_f32(vals[0] - vals[1]) if len(vals) > 1 else 0.0


def magic_run_ladder(state0, lift, residual, theta1, t2, tau_v):
    """The margin ladder itself -- format-agnostic, matches magic.py's
    run_ladder() exactly (rung 1 REPLACE, rung 2 gated ADD)."""
    state = state0
    if magic_margin(state) <= theta1:
        state = lift
    if magic_margin(state) <= t2:
        cand = {c: state[c] + residual[c] for c in state}
        if magic_margin(cand) - magic_margin(state) >= tau_v:
            state = cand
    return state


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def has_mcq_options_from_messages(messages) -> bool:
    import re

    text = "\n".join(
        item.get("text", "")
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )
    return re.search(r"(?:^|\n)[A-D]\. .+(?:\n[A-D]\. .+)+", text) is not None


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map

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


def mean_rgb(image: Image.Image) -> tuple[int, int, int]:
    small = image.convert('RGB').resize((1, 1))
    return tuple(int(v) for v in small.getpixel((0, 0)))


def patch_mask_image(
    image: Image.Image,
    patch_size: int = 32,
    mask_ratio: float = 0.35,
    seed: int = 42,
) -> Image.Image:
    """Mask random image patches with the image mean color."""
    import random
    from PIL import ImageDraw

    image = image.convert('RGB')
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    w, h = image.size
    boxes = [
        (x, y, min(x + patch_size, w), min(y + patch_size, h))
        for y in range(0, h, patch_size)
        for x in range(0, w, patch_size)
    ]
    if not boxes:
        return masked
    rng = random.Random(seed + w * 1009 + h * 9173)
    n_mask = max(1, round(len(boxes) * mask_ratio))
    fill = mean_rgb(image)
    for box in rng.sample(boxes, min(n_mask, len(boxes))):
        draw.rectangle(box, fill=fill)
    return masked


def salient_patch_mask_image(
    image: Image.Image,
    grid_size: int = 16,
    mask_ratio: float = 0.25,
) -> Image.Image:
    """Mask high-edge grid cells as a lightweight salient/object-like counterfactual."""
    from PIL import ImageDraw, ImageFilter

    image = image.convert('RGB')
    masked = image.copy()
    w, h = image.size
    if w <= 0 or h <= 0:
        return masked

    edge = image.convert('L').filter(ImageFilter.FIND_EDGES).resize((grid_size, grid_size))
    scores = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            scores.append((edge.getpixel((gx, gy)), gx, gy))
    scores.sort(reverse=True)
    n_mask = max(1, round(len(scores) * mask_ratio))

    draw = ImageDraw.Draw(masked)
    fill = mean_rgb(image)
    for _, gx, gy in scores[:n_mask]:
        left = round(gx * w / grid_size)
        upper = round(gy * h / grid_size)
        right = round((gx + 1) * w / grid_size)
        lower = round((gy + 1) * h / grid_size)
        draw.rectangle((left, upper, right, lower), fill=fill)
    return masked


def multiscale_blur_image(image: Image.Image) -> Image.Image:
    """Remove fine details by downsampling, upsampling, then blurring."""
    from PIL import ImageFilter

    image = image.convert('RGB')
    w, h = image.size
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    small = image.resize((max(1, w // 4), max(1, h // 4)), resample=resample)
    restored = small.resize((w, h), resample=resample)
    return restored.filter(ImageFilter.GaussianBlur(radius=20))


def center_mask_image(image: Image.Image, keep_context: float = 0.5) -> Image.Image:
    """Mask the center region, leaving surrounding context intact."""
    from PIL import ImageDraw

    image = image.convert('RGB')
    masked = image.copy()
    w, h = image.size
    box_w = round(w * keep_context)
    box_h = round(h * keep_context)
    left = (w - box_w) // 2
    upper = (h - box_h) // 2
    fill = mean_rgb(image)
    ImageDraw.Draw(masked).rectangle((left, upper, left + box_w, upper + box_h), fill=fill)
    return masked


def border_mask_image(image: Image.Image, keep_center: float = 0.5) -> Image.Image:
    """Mask border/context regions, leaving the image center intact."""
    from PIL import ImageDraw

    image = image.convert('RGB')
    masked = image.copy()
    w, h = image.size
    center_w = round(w * keep_center)
    center_h = round(h * keep_center)
    left = (w - center_w) // 2
    upper = (h - center_h) // 2
    right = left + center_w
    lower = upper + center_h
    fill = mean_rgb(image)
    draw = ImageDraw.Draw(masked)
    draw.rectangle((0, 0, w, upper), fill=fill)
    draw.rectangle((0, lower, w, h), fill=fill)
    draw.rectangle((0, upper, left, lower), fill=fill)
    draw.rectangle((right, upper, w, lower), fill=fill)
    return masked



def prompt_variation1(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '结合问题与选项仔细观察图像中的信息，请直接回答选项字母。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'Think about the question based on details in the given image. Please select the correct answer from the options above.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'Think about the question based on details in the given image. Please answer yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Think about the question based on details in the given image. Please try to answer the question with short words or phrases if possible.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'Think about the question based on details in the given image. Answer the question directly using a single word or phrase.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation2(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', 'Please carefully examine the information in the image, then consider the question and options, and reply directly with the letter corresponding to the correct answer from the options above.'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', '请仔细观察图像中的信息，然后结合问题与选项，从上述所有选项中直接回答正确选项对应的字母。'))
        elif 'Please answer yes or no.' in text_item:
            #new_texts.append(text_item.replace('Please answer yes or no.', '请直接回答yes或no。'))
            new_texts.append(text_item.replace('Please answer yes or no.', '观察给出的图片，请直接回答yes或no。'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_option_shuffle(texts, seed=42):
    """TCF variant: shuffle the A/B/C/D option order to expose position bias.

    Parses options from MCQ prompts, randomly reorders them, and records
    the permutation in the returned text as a hidden comment so evaluation
    code can recover the mapping if needed.  The ground-truth label token
    follows the permuted label in the shuffled prompt.
    """
    import re, random
    rng = random.Random(seed)
    new_texts = []
    for text_item in texts:
        # Match "A. ...\nB. ...\nC. ...\nD. ..." blocks
        option_pat = re.compile(
            r'((?:[A-D]\. .+\n?)+)',
            re.MULTILINE,
        )
        match = option_pat.search(text_item)
        if match is None:
            new_texts.append(text_item)
            continue
        block = match.group(0)
        line_pat = re.compile(r'^([A-D])\. (.+)$', re.MULTILINE)
        options = line_pat.findall(block)  # [('A', 'text'), ('B', 'text'), ...]
        if len(options) < 2:
            new_texts.append(text_item)
            continue
        labels = [o[0] for o in options]
        contents = [o[1] for o in options]
        perm = list(range(len(contents)))
        rng.shuffle(perm)
        shuffled_block = '\n'.join(
            f'{labels[i]}. {contents[perm[i]]}' for i in range(len(labels))
        ) + '\n'
        new_text = text_item[:match.start()] + shuffled_block + text_item[match.end():]
        new_texts.append(new_text)
    return new_texts


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

    if re.search(r'[\u4e00-\u9fff]', segment):
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


def prompt_options_only(texts):
    """TCF prior probe: remove the real question and keep answer options."""
    import re

    option_pat = re.compile(r'((?:[A-D]\. .+\n?)+)', re.MULTILINE)
    new_texts = []
    for text_item in texts:
        match = option_pat.search(text_item)
        if match is not None:
            body_start = _prompt_body_start(text_item, match.start())
            prefix = text_item[:body_start]
            replacement = '\nQuestion content is intentionally hidden. Choose the most plausible answer from the options only.\n'
            new_texts.append(prefix + replacement + text_item[match.start():])
            continue

        instr_start = _instruction_start(text_item)
        if instr_start >= 0:
            body_start = _prompt_body_start(text_item, instr_start)
            prefix = text_item[:body_start]
            replacement = '\nQuestion content is intentionally hidden. Use only the answer format prior. '
            new_texts.append(prefix + replacement + text_item[instr_start:])
            continue

        new_texts.append(text_item)
    return new_texts


def prompt_question_mask(texts):
    """TCF grounding probe: mask question content while preserving format."""
    import re

    option_pat = re.compile(r'((?:[A-D]\. .+\n?)+)', re.MULTILINE)
    new_texts = []
    for text_item in texts:
        match = option_pat.search(text_item)
        if match is not None:
            body_start = _prompt_body_start(text_item, match.start())
            segment = text_item[body_start:match.start()]
            masked = _mask_question_words(segment)
            if masked == segment:
                masked = '\nThe key question words are intentionally masked.\n'
            new_texts.append(text_item[:body_start] + masked + text_item[match.start():])
            continue

        instr_start = _instruction_start(text_item)
        if instr_start >= 0:
            body_start = _prompt_body_start(text_item, instr_start)
            segment = text_item[body_start:instr_start]
            masked = _mask_question_words(segment)
            if masked == segment:
                masked = '\nThe key question words are intentionally masked. '
            new_texts.append(text_item[:body_start] + masked + text_item[instr_start:])
            continue

        new_texts.append(text_item)
    return new_texts


def prompt_paraphrase1(texts):
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '请根据图像内容判断正确选项，并且只输出对应的选项字母。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'Use the image to choose the correct option. Respond with only the option letter.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'Use the image to decide the answer. Respond only with yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Use the image to answer concisely, preferably with a short word or phrase.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'Use the image to answer directly with one word or a short phrase.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_answer_format1(texts):
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '请只输出最终答案的选项字母，不要输出解释。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'Return only the final option letter, with no explanation.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'Return exactly one word: yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Return only the final short answer, without a full sentence.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'Return only the final short answer, without a full sentence.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation3(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '你是一名擅长回答选择题的聪明学生，请直接回答选项字母。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'You are a smart student who is good at answering multiple-choice questions. Please select the correct answer from the options above.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'You are a smart student who is good at answering yes or no questions. Please answer yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'You are a smart student who is good at answering questions. Please try to answer the question with short words or phrases if possible.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'You are a smart student who is good at answering questions. Answer the question directly using a single word or phrase.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts



class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
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
        beta_high=0.2,
        adaptive_threshold=0.0617,
        magic_theta1=0.0,
        magic_t2=0.0,
        magic_tau_v=0.0,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        probe_capture: bool = False,
        probe_path: str | None = None,
        probe_hidden_layers: tuple[int, ...] | list[int] = (-1, -8, -16),
        probe_attentions: bool = False,
        probe_attn_implementation: str | None = None,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        # custom parameters
        self.visual_type = visual_type
        self.textual_type = textual_type
        print(f"==> Using Image type: {visual_type}")
        print(f"==> Using Prompt type: {textual_type}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.beta_high = beta_high
        self.adaptive_threshold = adaptive_threshold
        self.magic_theta1 = magic_theta1
        self.magic_t2 = magic_t2
        self.magic_tau_v = magic_tau_v

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if self.visual_type in SCI_VISUAL_TYPES:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,   # <‑‑ ask for a dict‑like GenerationOutput
                output_logits=True,
            )
        else:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        if probe_capture:
            self.generate_kwargs['output_hidden_states'] = True
            self.generate_kwargs['return_dict_in_generate'] = True
            if probe_attentions:
                self.generate_kwargs['output_attentions'] = True
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLProcessor
            from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        attn_impl = probe_attn_implementation or 'flash_attention_2'

        if '72b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map=split_model(), attn_implementation=attn_impl
            )
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map='auto', attn_implementation=attn_impl
            )
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map='cpu', attn_implementation=attn_impl
            )
            self.model.cuda().eval()

        # Kaihua Modified
        if save_logits and (model_name is not None) and (dump_path is not None):
            save_path = os.path.join(dump_path, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Saving logit tensors to {save_path}")
            self.model.dump_path = save_path
        else:
            self.model.dump_path = None

        self.model.probe_path = probe_path if probe_capture else None
        self.model.probe_hidden_layers = tuple(probe_hidden_layers)
        self.model.probe_attentions = probe_attentions


        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_ids_or_logits(self, messages, visual_type, textual_type, cf_logits=None, cf_params=None, get_logits=False):
        texts = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])

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
        elif visual_type == 'vcf_patch_shuffle':
            # On-manifold VCF: divide into patches and shuffle with fixed seed
            import random
            patch_size = 32
            for image in images:
                w, h = image.size
                pw, ph = w // patch_size, h // patch_size
                patches = [image.crop((j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size))
                           for i in range(ph) for j in range(pw)]
                rng = random.Random(42)
                rng.shuffle(patches)
                shuffled = Image.new("RGB", (pw * patch_size, ph * patch_size))
                for idx, patch in enumerate(patches):
                    shuffled.paste(patch, ((idx % pw) * patch_size, (idx // pw) * patch_size))
                processed_images.append(shuffled)
        elif visual_type == 'vcf_blur':
            # Gaussian blur — preserves layout, removes fine-grained semantic content
            from PIL import ImageFilter
            for image in images:
                processed_images.append(image.filter(ImageFilter.GaussianBlur(radius=50)))
        elif visual_type == 'vcf_strong_blur':
            # Stronger Gaussian blur — stress test for residual fine-detail reliance
            from PIL import ImageFilter
            for image in images:
                processed_images.append(image.filter(ImageFilter.GaussianBlur(radius=100)))
        elif visual_type == 'vcf_multiscale_blur':
            # Multi-scale blur — removes high frequency detail while keeping coarse layout
            for image in images:
                processed_images.append(multiscale_blur_image(image))
        elif visual_type == 'vcf_patch_mask':
            # Random patch masking — partial visual evidence removal while preserving color prior
            for image in images:
                processed_images.append(patch_mask_image(image))
        elif visual_type == 'vcf_salient_mask':
            # Salient/object-like masking — remove high-edge regions as a detector-free proxy
            for image in images:
                processed_images.append(salient_patch_mask_image(image))
        elif visual_type == 'vcf_center_mask':
            # Center mask — removes common object-centered evidence, keeps context
            for image in images:
                processed_images.append(center_mask_image(image))
        elif visual_type == 'vcf_border_mask':
            # Border mask — removes contextual border evidence, keeps center
            for image in images:
                processed_images.append(border_mask_image(image))
        elif visual_type == 'vcf_grayscale':
            # Remove color information, keep luminance structure
            for image in images:
                processed_images.append(image.convert('L').convert('RGB'))
        elif visual_type == 'vcf_high_pass':
            # High-pass — remove LOW frequency (smooth color/luminance regions), keep edges/high freq.
            # Frequency-complement of vcf_blur (low-pass r=50): tests the orthogonal half of the
            # frequency axis the existing blur channels never probe.
            from PIL import ImageFilter, ImageChops
            for image in images:
                low = image.filter(ImageFilter.GaussianBlur(radius=50))
                processed_images.append(ImageChops.subtract(image, low, scale=1.0, offset=128))
        elif visual_type == 'vcf_phase_scramble':
            # Phase scramble — randomize Fourier PHASE, keep MAGNITUDE. Preserves power spectrum
            # (texture statistics) and color histogram, destroys global shape/structure. Probes the
            # shape-vs-texture axis — orthogonal to both color (grayscale) and frequency (blur/high-pass).
            import numpy as np
            for image in images:
                arr = np.asarray(image.convert('RGB')).astype(np.float32)
                rng = np.random.RandomState(42)
                out = np.empty_like(arr)
                for ch in range(3):
                    f = np.fft.fft2(arr[:, :, ch])
                    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, f.shape))
                    out[:, :, ch] = np.real(np.fft.ifft2(np.abs(f) * ph))
                out = np.clip(out, 0, 255).astype(np.uint8)
                processed_images.append(Image.fromarray(out))
        elif visual_type == 'vcf_hue_shift':
            # Hue shift by 180deg — CHANGE color (complementary hue) while keeping luminance, structure,
            # saturation. The other pole of the color axis (grayscale REMOVES color; this CORRUPTS it).
            for image in images:
                h, s, v = image.convert('HSV').split()
                h = h.point(lambda x: (x + 128) % 256)
                processed_images.append(Image.merge('HSV', (h, s, v)).convert('RGB'))
        else:
            raise ValueError("Wrong Image Type")
        
        # process text
        processed_texts = []
        if textual_type in ('default', *SCI_VISUAL_TYPES):
            processed_texts = texts
        elif textual_type == 'tcf_v1':
            processed_texts = prompt_variation1(texts)
        elif textual_type == 'tcf_v2':
            processed_texts = prompt_variation2(texts)
        elif textual_type == 'tcf_v3':
            processed_texts = prompt_variation3(texts)
        elif textual_type == 'tcf_option_shuffle1':
            processed_texts = prompt_option_shuffle(texts, seed=42)
        elif textual_type == 'tcf_option_shuffle2':
            processed_texts = prompt_option_shuffle(texts, seed=137)
        elif textual_type == 'tcf_paraphrase1':
            processed_texts = prompt_paraphrase1(texts)
        elif textual_type == 'tcf_answer_format1':
            processed_texts = prompt_answer_format1(texts)
        elif textual_type == 'tcf_options_only':
            processed_texts = prompt_options_only(texts)
        elif textual_type == 'tcf_question_mask':
            processed_texts = prompt_question_mask(texts)
        else:
            raise ValueError("Wrong Text Type")
        
        # inference
        if visual_type in SCI_VISUAL_TYPES:
            assert (cf_logits is not None) and (cf_params is not None)
            inputs = self.processor(text=processed_texts, images=processed_images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
            output_dicts = self.model.generate(
                    cf_logits = cf_logits,
                    cf_params = cf_params,
                    **inputs,
                    **self.generate_kwargs,
                )
            logits = torch.cat(output_dicts.logits, dim=0)
            generated_ids = logits.unsqueeze(0).argmax(dim=-1)
            return generated_ids
        else:
            # Original Code
            inputs = self.processor(text=processed_texts, images=processed_images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
            if get_logits:
                output_dicts = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
                logits = torch.cat(output_dicts.logits, dim=0)
                return logits.unsqueeze(0).float()
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
                if hasattr(generated_ids, 'sequences'):
                    generated_ids = generated_ids.sequences
                # zip() pairs one prompt against one output, which silently drops
                # all but the first sequence when num_return_sequences > 1 (the
                # beam5 pools the open-ended path needs).  Slice by prompt length
                # instead; identical for the single-sequence case.
                prompt_len = inputs.input_ids.shape[1]
                generated_ids = [output_ids[prompt_len:] for output_ids in generated_ids]
                return generated_ids

    def generate_inner(self, message, dataset=None):
        

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if self.visual_type == 'TIE':
            # We inplement TIE based on paper Counterfactual VQA and Unbiased Scene Graph Generation
            tie_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"tie_logits": tie_logits}
            cf_params = {"theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'VCD':
            # Visual Contrastive Decoding (VCD)
            # Current BS-Subsets only predict one character or one word, so we don't need to take care of auto-regressive generation, focusing on the first token is enough. 
            # It can be generalized to iterative generation in future work if needed.
            vcd_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"vcd_logits": vcd_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'M3ID':
            m3id_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"m3id_logits": m3id_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI3':
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "tcf1_logits": tcf1_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI5':
            # The proposed self-critical inference
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in SCI_SINGLE_VCF_SOURCES:
            vcf_logits = self.generate_ids_or_logits(
                messages,
                visual_type=SCI_SINGLE_VCF_SOURCES[self.visual_type],
                textual_type='default',
                get_logits=True,
            )
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"vcf1_logits": vcf_logits, "vcf2_logits": vcf_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Fallback':
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_strong_blur', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_center_mask', textual_type='default', get_logits=True)
            if has_mcq_options_from_messages(messages):
                tcf1_type, tcf2_type = 'tcf_option_shuffle1', 'tcf_option_shuffle2'
            else:
                tcf1_type, tcf2_type = 'tcf_question_mask', 'tcf_options_only'
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type=tcf1_type, get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type=tcf2_type, get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Fallback-OOD':
            # Option-2 floor: clean format-aware TC probes (same as Fallback) but VC = FULL
            # ablation (black + noise) instead of partial blur/mask.
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            if has_mcq_options_from_messages(messages):
                tcf1_type, tcf2_type = 'tcf_option_shuffle1', 'tcf_option_shuffle2'
            else:
                tcf1_type, tcf2_type = 'tcf_question_mask', 'tcf_options_only'
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type=tcf1_type, get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type=tcf2_type, get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type in CLEAN_JOINT_SOURCES:
            visual_source, text_source = CLEAN_JOINT_SOURCES[self.visual_type]
            vcf_logits = self.generate_ids_or_logits(messages, visual_type=visual_source, textual_type='default', get_logits=True)
            tcf_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type=text_source, get_logits=True)
            cf_logits = {"vcf1_logits": vcf_logits, "tcf1_logits": tcf_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI7':
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            vcf3_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise400', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            tcf3_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v3', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "vcf3_logits": vcf3_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits, "tcf3_logits": tcf3_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Ablation1':
            ablation_vcf1 = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            ablation_vcf2 = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            ablation_tcf1 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            cf_logits = {"ablation_vcf1": ablation_vcf1, "ablation_vcf2": ablation_vcf2, "ablation_tcf1": ablation_tcf1}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Ablation2':
            ablation_vcf1 = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            ablation_tcf1 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            ablation_tcf2 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"ablation_vcf1": ablation_vcf1, "ablation_tcf1": ablation_tcf1, "ablation_tcf2": ablation_tcf2}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI-Adaptive':
            # Adaptive routing: image_info_conf determines VCF type and beta.
            # Low image_info_conf → Fixed (OOD VCF, beta=0.1)   — 5 passes total (same as SCI5)
            # High image_info_conf → Action B (Shuffle VCF, beta=0.2) — 6 passes total
            #
            # Optimization: orig_logits are reused for BOTH routing (image_info_conf) AND the SCI
            # formula, eliminating the redundant generation forward pass that SCI5 would add.
            orig_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='default', get_logits=True)
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)

            # Extract first-token logits [1, vocab] for routing and SCI formula
            o  = orig_logits[:, 0, :].float()
            v1 = vcf1_logits[:, 0, :].float()
            v2 = vcf2_logits[:, 0, :].float()
            t1 = tcf1_logits[:, 0, :].float()
            t2 = tcf2_logits[:, 0, :].float()

            # Routing signal: image_info_conf = max(softmax(orig)) - max(softmax(ood_mean))
            ood_mean = (v1 + v2) / 2.0
            orig_conf = torch.softmax(o, dim=-1).max(dim=-1).values.mean().item()
            ood_conf  = torch.softmax(ood_mean, dim=-1).max(dim=-1).values.mean().item()
            image_info_conf = orig_conf - ood_conf

            adaptive_threshold = getattr(self, 'adaptive_threshold', 0.0617)
            if image_info_conf > adaptive_threshold:
                # Action B: Shuffle-only VCF + beta_high (1 extra pass)
                vs = self.generate_ids_or_logits(messages, visual_type='vcf_patch_shuffle', textual_type='default', get_logits=True)[:, 0, :].float()
                vcf_mean = vs
                beta = self.beta_high
            else:
                # Fixed: OOD VCF mean + conservative beta (no extra pass)
                vcf_mean = ood_mean
                beta = self.beta

            # Apply SCI formula directly using cached orig — no generation forward pass needed.
            # Mirrors modeling_qwen2_vl.py get_consistent_logits_3 (confidence type='constant'→1.0)
            alpha, gamma, theta = self.alpha, self.gamma, self.theta
            consistent_logits = torch.cat([o, t1, t2], dim=0).max(dim=0).values / gamma  # [vocab]
            unbiased_weights  = (o - vcf_mean) / beta                                    # [1, vocab]
            unbiased_logits   = consistent_logits + unbiased_weights                      # [1, vocab]
            cutoff = torch.log(torch.tensor([theta], device=o.device)) + consistent_logits.max(dim=-1, keepdim=True).values
            final_logits = unbiased_logits.masked_fill(consistent_logits < cutoff, float('-inf'))
            generated_ids = final_logits.argmax(dim=-1).unsqueeze(0)  # [1, 1]
        elif self.visual_type == 'MAGIC':
            # Live MAGIC: the same margin ladder magic.py replays offline
            # (run_ladder), embedded here the way SCI-Adaptive embeds SCI's
            # formula -- one call, internally multi-pass, single argmax out.
            # MCQ only: needs the option letters materialized in the prompt
            # to reduce full-vocab logits down to a per-candidate margin,
            # exactly like build_mcq_bundle.py's offline reduction does.
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

            real_def = self.generate_ids_or_logits(messages, visual_type='default', textual_type='default', get_logits=True)[:, 0, :].float()
            blank_def = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)[:, 0, :].float()
            real_ans = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_answer_format1', get_logits=True)[:, 0, :].float()
            blank_ans = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='tcf_answer_format1', get_logits=True)[:, 0, :].float()
            gray = self.generate_ids_or_logits(messages, visual_type='vcf_grayscale', textual_type='default', get_logits=True)[:, 0, :].float()
            strongblur = self.generate_ids_or_logits(messages, visual_type='vcf_strong_blur', textual_type='default', get_logits=True)[:, 0, :].float()
            centermask = self.generate_ids_or_logits(messages, visual_type='vcf_center_mask', textual_type='default', get_logits=True)[:, 0, :].float()

            # Reduce each full-vocab view down to per-option-letter scores.
            s0 = magic_reduce_to_letters(real_def[0], letter_ids)
            bd = magic_reduce_to_letters(blank_def[0], letter_ids)
            ra = magic_reduce_to_letters(real_ans[0], letter_ids)
            ba = magic_reduce_to_letters(blank_ans[0], letter_ids)
            default_delta = {c: s0[c] - bd[c] for c in s0}
            answer_delta = {c: ra[c] - ba[c] for c in s0}
            lift = {c: max(default_delta[c], answer_delta[c]) for c in s0}

            # Route the visual-residual channel by argmin image_info_conf,
            # the same live signal SCI-Adaptive computes for its own routing
            # (max-softmax confidence gap between orig and the corrupted view).
            orig_conf = torch.softmax(real_def[0], dim=-1).max().item()
            channels = {'gray': gray, 'strongblur': strongblur, 'centermask': centermask}
            iic = {name: orig_conf - torch.softmax(v[0], dim=-1).max().item() for name, v in channels.items()}
            routed = min(iic, key=iic.get)
            vc = magic_reduce_to_letters(channels[routed][0], letter_ids)
            residual = {c: s0[c] - vc[c] for c in s0}

            state = magic_run_ladder(s0, lift, residual, self.magic_theta1, self.magic_t2, self.magic_tau_v)
            pred_letter = max(state, key=state.get)
            generated_ids = torch.tensor([letter_ids[pred_letter][:1]], device=real_def.device)
        else:
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type)


        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
