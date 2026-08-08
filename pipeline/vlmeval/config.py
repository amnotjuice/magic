from vlmeval.vlm import *
from functools import partial
import os

save_logits = True
dump_path = os.environ.get("MAGIC_DUMP_PATH", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "dump_tensors"))

qwen2vl_custom_series = {
    "Qwen2-VL-7B-Original": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Original",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "Qwen2-VL-7B-Original-Probe": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_new_tokens=1,
        model_name="Qwen2-VL-7B-Original-Probe",
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="default",
        probe_capture=True,
        probe_path="/data/shunshungu/Self-Critical-Inference-Framework/dump_tensors_probe/Qwen2-VL-7B-Original-Probe",
        probe_hidden_layers=(-1, -8, -16),
        probe_attentions=False,
    ),

    "Qwen2-VL-7B-Original-ProbeAttn": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_new_tokens=1,
        model_name="Qwen2-VL-7B-Original-ProbeAttn",
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="default",
        probe_capture=True,
        probe_path="/data/shunshungu/Self-Critical-Inference-Framework/dump_tensors_probe/Qwen2-VL-7B-Original-ProbeAttn",
        probe_hidden_layers=(-1, -8, -16),
        probe_attentions=True,
        probe_attn_implementation="eager",
    ),

    "Qwen2-VL-7B-VCF-Color0": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Color0",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_color0",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "Qwen2-VL-7B-VCF-Color255": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Color255",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_color255",    # default, vcf_color0, vcf_color255, vcf_noise400, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "Qwen2-VL-7B-VCF-Noise400": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Noise400",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_noise400",    # default, vcf_color0, vcf_color255, vcf_noise400, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "Qwen2-VL-7B-VCF-Noise500": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Noise500",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_noise500",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    # --- New VCF types (on-manifold and alternative counterfactuals) ---
    "Qwen2-VL-7B-VCF-Shuffle": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Shuffle",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_patch_shuffle",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-Blur": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Blur",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_blur",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-HighPass": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-HighPass",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_high_pass",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-HueShift": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-HueShift",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_hue_shift",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-PhaseScramble": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-PhaseScramble",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_phase_scramble",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-StrongBlur": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-StrongBlur",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_strong_blur",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-MultiScaleBlur": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-MultiScaleBlur",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_multiscale_blur",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-PatchMask": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-PatchMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_patch_mask",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-SalientMask": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-SalientMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_salient_mask",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-CenterMask": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-CenterMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_center_mask",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-BorderMask": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-BorderMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_border_mask",
        textual_type="default",
    ),
    "Qwen2-VL-7B-VCF-Grayscale": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCF-Grayscale",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_grayscale",
        textual_type="default",
    ),
    # --- New TCF types (option-order shuffle for position bias) ---
    "Qwen2-VL-7B-TCF-OptionShuffle1": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-OptionShuffle1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_option_shuffle1",
    ),
    "Qwen2-VL-7B-TCF-OptionShuffle2": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-OptionShuffle2",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_option_shuffle2",
    ),
    "Qwen2-VL-7B-TCF-Paraphrase1": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-Paraphrase1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_paraphrase1",
    ),
    "Qwen2-VL-7B-TCF-AnswerFormat1": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-AnswerFormat1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_answer_format1",
    ),
    "Qwen2-VL-7B-TCF-OptionsOnly": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-OptionsOnly",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_options_only",
    ),
    "Qwen2-VL-7B-TCF-QuestionMask": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-QuestionMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_question_mask",
    ),
    # --- Adaptive SCI: routes per sample based on image_info_conf ---
    "Qwen2-VL-7B-SCI-Adaptive": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI-Adaptive",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI-Adaptive",
        textual_type="SCI-Adaptive",
        beta=0.1,       # conservative beta for low image_info_conf samples
        beta_high=0.2,  # aggressive beta for high image_info_conf samples
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
        adaptive_threshold=0.030979037284851,  # val-selected fixed-vs-shuffle router
    ),

    # --- answer-format views, added 2026-07-17 to make MAGIC's rung-1 answer
    # estimate reachable from the OFFICIAL run.py path (previously it existed
    # only inside research probes, which deviated from SCI's protocol).
    # textual_type="tcf_answer_format1" is vlmeval's own transform
    # (vlmeval/vlm/qwen2_vl/model.py:790); every other field mirrors the
    # sibling configs verbatim.
    "Qwen2-VL-7B-TCF-AnswerFormat": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-AnswerFormat",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="tcf_answer_format1",
    ),

    "Qwen2-VL-7B-TCF-AnswerFormat-Blank": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-AnswerFormat-Blank",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="vcf_color0",
        textual_type="tcf_answer_format1",
    ),

    "Qwen2-VL-7B-TCF-V1": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-V1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="tcf_v1",   # default, tcf_v1, tcf_v2
    ),
    
    "Qwen2-VL-7B-TCF-V2": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-V2",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="tcf_v2",   # default, tcf_v1, tcf_v2
    ),

    "Qwen2-VL-7B-TCF-V3": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TCF-V3",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.bfloat16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="tcf_v3",   # default, tcf_v1, tcf_v2, tcf_v3
    ),

    ############### Inference Algorithms ###############
    "Qwen2-VL-7B-TIE": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-TIE",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="TIE",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="default",   # default, tcf_v1, tcf_v2
        theta=0.5,
    ),

    "Qwen2-VL-7B-M3ID": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-M3ID",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="M3ID",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="default",   # default, tcf_v1, tcf_v2
        alpha=0.02,    
        theta=0.3,
    ),

    "Qwen2-VL-7B-VCD": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-VCD",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="VCD",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="default",   # default, tcf_v1, tcf_v2
        alpha=1.0,
        theta=0.3,
    ),


    ############### Self-Critical Inference Algorithms ###############
    "Qwen2-VL-7B-SCI3-b02a1g15t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI3",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI3",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI3",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=1.5,
        theta=0.3,
    ),

    "Qwen2-VL-7B-SCI5-b02a1g2t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    "Qwen2-VL-7B-MAGIC": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-MAGIC",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="MAGIC",
        textual_type="MAGIC",
        # frozen ladder constants, LADDER_MCQ["qwen"] in magic.py -- MCQ only,
        # live at generation time (see the MAGIC branch in vlm/qwen2_vl/model.py)
        magic_theta1=3.25,
        magic_t2=0.4,
        magic_tau_v=0.625,
    ),

    "Qwen2-VL-7B-SCI7-b02a1g25t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI7",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI7",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI7",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.3,
    ),   

    "Qwen2-VL-7B-SCI5-b01a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5-b01a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5",
        textual_type="SCI5",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-SCI5-StrongBlur-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5-StrongBlur-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5-StrongBlur",
        textual_type="SCI5",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-SCI5-CenterMask-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5-CenterMask-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5-CenterMask",
        textual_type="SCI5",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-Fallback-b01a1g2t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Fallback-b01a1g2t03",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Fallback",
        textual_type="Fallback",
        beta=0.1,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    "Qwen2-VL-7B-Fallback-OOD-b01a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Fallback-OOD-b01a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Fallback-OOD",
        textual_type="Fallback",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-Joint-StrongBlur-OptionShuffle1-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Joint-StrongBlur-OptionShuffle1-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Joint-StrongBlur-OptionShuffle1",
        textual_type="Joint-StrongBlur-OptionShuffle1",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-Joint-CenterMask-OptionShuffle1-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Joint-CenterMask-OptionShuffle1-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Joint-CenterMask-OptionShuffle1",
        textual_type="Joint-CenterMask-OptionShuffle1",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-Joint-StrongBlur-QuestionMask-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Joint-StrongBlur-QuestionMask-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Joint-StrongBlur-QuestionMask",
        textual_type="Joint-StrongBlur-QuestionMask",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-Joint-CenterMask-QuestionMask-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Joint-CenterMask-QuestionMask-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Joint-CenterMask-QuestionMask",
        textual_type="Joint-CenterMask-QuestionMask",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "Qwen2-VL-7B-SCI5-Grayscale-b02a1g25t02": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5-Grayscale-b02a1g25t02",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5-Grayscale",
        textual_type="SCI5",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),


    "Qwen2-VL-7B-SCI5-b02a1g2t08": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.0,
        theta=0.8,
    ), 


    "Qwen2-VL-7B-SCI-Ablation1": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Ablation1",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Ablation1",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="Ablation1",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=1.5,
        theta=0.3,
    ),


    "Qwen2-VL-7B-SCI-Ablation2": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-Ablation2",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="Ablation2",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="Ablation2",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    # [(0.3, 1.0, 1, 0.6)]
    # [(0.1, 1.0, 1.5, 0.7)]
    # [(0.2, 1.0, 2, 0.8)]

    "Qwen2-VL-7B-SCI5-b1a1g2t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=1.0,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    "Qwen2-VL-7B-SCI5-b05a1g2t03": partial(
        Qwen2VLChat,
        model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        model_name="Qwen2-VL-7B-SCI5",
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500, VCD
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=0.5,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

}

llava_custom_series = {
    "LLaVA-NeXT-8B-Original": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Original",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-VCF-Color0": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-Color0",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_color0",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-VCF-Color255": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-Color255",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_color255",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-VCF-Noise400": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-Noise400",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_noise400",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-VCF-Noise500": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-Noise500",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_noise500",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-VCF-StrongBlur": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-StrongBlur",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_strong_blur",
        textual_type="default",
    ),
    "LLaVA-NeXT-8B-VCF-HighPass": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-HighPass",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_high_pass",
        textual_type="default",
    ),

    "LLaVA-NeXT-8B-VCF-CenterMask": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-CenterMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_center_mask",
        textual_type="default",
    ),

    # --- answer-format views (see the Qwen note above).  dtype=float16 per
    # SCI 4.2 ("LLaVA-NeXT used float16 and greedy decoding").
    "LLaVA-NeXT-8B-TCF-AnswerFormat": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-AnswerFormat",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_answer_format1",
    ),

    "LLaVA-NeXT-8B-TCF-AnswerFormat-Blank": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-AnswerFormat-Blank",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_color0",
        textual_type="tcf_answer_format1",
    ),

    "LLaVA-NeXT-8B-TCF-V1": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-V1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="tcf_v1",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-TCF-V2": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-V2",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",    # default, vcf_color0, vcf_noise500
        textual_type="tcf_v2",   # default, tcf_v1, tcf_v2
    ),

    "LLaVA-NeXT-8B-TCF-V3": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-V3",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_v3",
    ),

    "LLaVA-NeXT-8B-TCF-OptionShuffle1": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-OptionShuffle1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_option_shuffle1",
    ),

    "LLaVA-NeXT-8B-TCF-OptionShuffle2": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-OptionShuffle2",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_option_shuffle2",
    ),
    "LLaVA-NeXT-8B-VCF-Grayscale": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCF-Grayscale",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="vcf_grayscale",
        textual_type="default",
    ),
    "LLaVA-NeXT-8B-TCF-QuestionMask": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-QuestionMask",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_question_mask",
    ),
    "LLaVA-NeXT-8B-TCF-OptionsOnly": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-OptionsOnly",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_options_only",
    ),
    "LLaVA-NeXT-8B-TCF-Paraphrase1": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-Paraphrase1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_paraphrase1",
    ),
    "LLaVA-NeXT-8B-TCF-AnswerFormat1": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TCF-AnswerFormat1",
        save_logits=save_logits,
        dump_path=dump_path,
        dtype=torch.float16,
        visual_type="default",
        textual_type="tcf_answer_format1",
    ),


    ############### Inference Algorithms ###############
    "LLaVA-NeXT-8B-TIE": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-TIE",
        dump_path=None,
        dtype=torch.float16,
        visual_type="TIE",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
        theta=0.5,
    ),

    "LLaVA-NeXT-8B-M3ID": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-M3ID",
        dump_path=None,
        dtype=torch.float16,
        visual_type="M3ID",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
        alpha=0.02,    
        theta=0.3,
    ),

    "LLaVA-NeXT-8B-VCD": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCD",
        dump_path=None,
        dtype=torch.float16,
        visual_type="VCD",    # default, vcf_color0, vcf_noise500
        textual_type="default",   # default, tcf_v1, tcf_v2
        alpha=1.0,
        theta=0.3,
    ),

    ############### Self-Critical Inference Algorithms ###############
    "LLaVA-NeXT-8B-SCI3-b02a1g15t03": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI3",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI3",    # default, vcf_color0, vcf_noise500
        textual_type="SCI3",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=1.5,
        theta=0.3,
    ),

    "LLaVA-NeXT-8B-MAGIC": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-MAGIC",
        dump_path=None,
        dtype=torch.float16,
        visual_type="MAGIC",
        textual_type="MAGIC",
        # frozen ladder constants, LADDER_MCQ["llava"] in magic.py -- MCQ only,
        # live at generation time (see the MAGIC branch in vlm/llava/llava.py)
        magic_theta1=5.03125,
        magic_t2=0.3,
        magic_tau_v=0.625,
    ),

    "LLaVA-NeXT-8B-SCI5-b02a1g2t03": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    "LLaVA-NeXT-8B-SCI5-b01a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5-b01a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5",
        textual_type="SCI5",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-SCI5-b05a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5-b05a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5",
        textual_type="SCI5",
        beta=0.5,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-SCI5-StrongBlur-b05a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5-StrongBlur-b05a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5-StrongBlur",
        textual_type="SCI5",
        beta=0.5,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-SCI5-CenterMask-b01a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5-CenterMask-b01a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5-CenterMask",
        textual_type="SCI5",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-Fallback-b01a1g2t03": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Fallback-b01a1g2t03",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Fallback",
        textual_type="Fallback",
        beta=0.1,
        alpha=1.0,
        gamma=2.0,
        theta=0.3,
    ),

    "LLaVA-NeXT-8B-Fallback-OOD-b01a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Fallback-OOD-b01a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Fallback-OOD",
        textual_type="Fallback",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-Joint-StrongBlur-OptionShuffle1-b02a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Joint-StrongBlur-OptionShuffle1-b02a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Joint-StrongBlur-OptionShuffle1",
        textual_type="Joint-StrongBlur-OptionShuffle1",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-Joint-CenterMask-OptionShuffle1-b02a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Joint-CenterMask-OptionShuffle1-b02a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Joint-CenterMask-OptionShuffle1",
        textual_type="Joint-CenterMask-OptionShuffle1",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-Joint-StrongBlur-QuestionMask-b02a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Joint-StrongBlur-QuestionMask-b02a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Joint-StrongBlur-QuestionMask",
        textual_type="Joint-StrongBlur-QuestionMask",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-Joint-CenterMask-QuestionMask-b02a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-Joint-CenterMask-QuestionMask-b02a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="Joint-CenterMask-QuestionMask",
        textual_type="Joint-CenterMask-QuestionMask",
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-VCStrong-b01a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCStrong-b01a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="VC-StrongBlur",
        textual_type="default",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-VCStrongExact-b01a1g25t02": partial(
        LLaVA_Next,
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-VCStrongExact-b01a1g25t02",
        dump_path=None,
        dtype=torch.float16,
        visual_type="VC-StrongBlur-Exact",
        textual_type="default",
        beta=0.1,
        alpha=1.0,
        gamma=2.5,
        theta=0.2,
    ),

    "LLaVA-NeXT-8B-SCI7-b02a1g25t03": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI7",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI7",    # default, vcf_color0, vcf_noise500
        textual_type="SCI7",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.5,
        theta=0.3,
    ),


    "LLaVA-NeXT-8B-SCI5-b02a1g2t08": partial(
        LLaVA_Next, 
        model_path="/data/shunshungu/models/llama3-llava-next-8b-hf",
        model_name="LLaVA-NeXT-8B-SCI5",
        dump_path=None,
        dtype=torch.float16,
        visual_type="SCI5",    # default, vcf_color0, vcf_noise500
        textual_type="SCI5",   # default, tcf_v1, tcf_v2
        beta=0.2,
        alpha=1.0,
        gamma=2.0,
        theta=0.8,
    ),


}

supported_VLM = {}
model_groups = [qwen2vl_custom_series, llava_custom_series]
for grp in model_groups:
    supported_VLM.update(grp)
