"""Shared action grammar for evidence-routed SCI experiments.

Most historical action names are cache column names or model-output folder names,
so executable aliases must remain stable. The clean main-method candidate set
uses short paper-facing names and keeps historical SCI/OOD actions in ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ActionSpec:
    family: str
    probe: str
    aggregation: str
    description: str

    @property
    def readable_name(self) -> str:
        return f"{self.family}:{self.probe}:{self.aggregation}"


ROUTE_DISPLAY_NAMES = {
    "floor": "conservative",
    "tc": "text_conflict",
    "tc_low": "text_conflict_low",
    "tc_high": "text_conflict_high",
    "vc": "visual_conflict",
    "both": "joint_conflict",
}

SCI_BOTH = "both_max_ood_b0.1_g2.5_t0.2"
SCI_OS_BOTH = "both_max_ood_os_b0.1_g2.5_t0.2"
OLD_CLEAN_FALLBACK = "fallback_2tc2vc_b0.1_g2_t0.3"
CLEAN_FALLBACK = "conservative_2tc2vc_b0.1_g2_t0.3"
LEGACY_FALLBACK = "floor_sci5_ours_aug_b0.1_g2_t0.3"

EXECUTABLE_ACTION_ALIASES = {
    CLEAN_FALLBACK: LEGACY_FALLBACK,
    OLD_CLEAN_FALLBACK: LEGACY_FALLBACK,
    "joint_strong_blur_option_shuffle1_b0.2_g2.5_t0.2": "Joint-StrongBlur-OptionShuffle1",
    "joint_center_mask_option_shuffle1_b0.2_g2.5_t0.2": "Joint-CenterMask-OptionShuffle1",
    "joint_strong_blur_question_mask_b0.2_g2.5_t0.2": "Joint-StrongBlur-QuestionMask",
    "joint_center_mask_question_mask_b0.2_g2.5_t0.2": "Joint-CenterMask-QuestionMask",
}

PUBLIC_ACTION_NAMES = {
    LEGACY_FALLBACK: "Conservative-2VC2TC",
    OLD_CLEAN_FALLBACK: "Conservative-2VC2TC",
    CLEAN_FALLBACK: "Conservative-2VC2TC",
    "b5_tc_os1_direct": "Text-OptionShuffle1",
    "tcg_qmask_direct": "Text-QuestionMask",
    "tcg_options_direct": "Text-OptionsOnly",
    "both_max_strong_blur_b0.2_g2.5_t0.2": "Visual-StrongBlur",
    "both_max_center_mask_b0.2_g2.5_t0.2": "Visual-CenterMask",
    "b5_joint_strong_os1": "Joint-StrongBlur-OptionShuffle1",
    "b5_joint_center_os1": "Joint-CenterMask-OptionShuffle1",
    "joint_os_strong_blur_b0.2_g2.5_t0.2": "Joint-StrongBlur-OptionShuffle1",
    "joint_os_center_mask_b0.2_g2.5_t0.2": "Joint-CenterMask-OptionShuffle1",
    "tcg_both_strong_qmask_b2_t0.05": "Joint-StrongBlur-QuestionMask",
    "tcg_both_center_qmask_b2_t0.05": "Joint-CenterMask-QuestionMask",
    "joint_strong_blur_option_shuffle1_b0.2_g2.5_t0.2": "Joint-StrongBlur-OptionShuffle1",
    "joint_center_mask_option_shuffle1_b0.2_g2.5_t0.2": "Joint-CenterMask-OptionShuffle1",
    "joint_strong_blur_question_mask_b0.2_g2.5_t0.2": "Joint-StrongBlur-QuestionMask",
    "joint_center_mask_question_mask_b0.2_g2.5_t0.2": "Joint-CenterMask-QuestionMask",
}

MAIN_VISUAL_SOURCES = ("strong_blur", "center_mask")
MAIN_VISUAL_STRENGTHS = ("fixed",)
MAIN_VC_PARAMS = {
    "fixed": (0.2, 2.5, 0.2),
}

SOURCE_VARIANTS = {
    "strong_blur": "VCF-StrongBlur",
    "center_mask": "VCF-CenterMask",
    "grayscale": "VCF-Grayscale",
}

SOURCE_FEATURES = [
    "image_info_conf_strong_blur",
    "image_info_conf_center_mask",
    "image_info_conf_grayscale",
    "d_vc_strong_blur_js",
    "d_vc_center_mask_js",
    "d_vc_grayscale_js",
    "vc_same_count_blur",
    "uncertainty_score",
]

ACTION_SPECS = {
    "p1_core": ActionSpec(
        "Visual",
        "strong_blur_confidence",
        "two_branch",
        "Legacy visual selector; use Visual(source,strength) in the clean main method.",
    ),
    "vc_ood_b0.1_t0.1": ActionSpec(
        "Ablation",
        "blank_noise",
        "fixed_beta",
        "Legacy blank/noise visual correction kept for ablation, not the clean main method.",
    ),
    "b5_tc_os1_direct": ActionSpec(
        "Text",
        "option_shuffle_1",
        "direct",
        "Re-asks with shuffled options and unshuffles the answer label.",
    ),
    "b5_tc_os2_direct": ActionSpec(
        "Conservative",
        "option_shuffle_2",
        "direct",
        "Second option-shuffle probe used only inside the conservative strategy.",
    ),
    "b5_tc_os_mean": ActionSpec(
        "TextConsistency",
        "option_shuffle",
        "mean",
        "Averages option-shuffle evidence.",
    ),
    "b5_tc_os_max": ActionSpec(
        "TextConsistency",
        "option_shuffle",
        "max",
        "Uses the strongest option-shuffle evidence.",
    ),
    "tc_optionshuffle2_unshuffle": ActionSpec(
        "TextConsistency",
        "option_shuffle_2",
        "raw_unshuffle",
        "Raw unshuffled answer from the second option-shuffle probe.",
    ),
    "tcg_qmask_direct": ActionSpec(
        "Text",
        "question_mask",
        "direct",
        "Generates from a question-masked text counterfactual.",
    ),
    "tcg_qmask_mean": ActionSpec(
        "TextConsistency",
        "question_mask",
        "mean",
        "Averages question-mask counterfactual evidence.",
    ),
    "tcg_qmask_max": ActionSpec(
        "TextConsistency",
        "question_mask",
        "max",
        "Uses the strongest question-mask counterfactual evidence.",
    ),
    "tcg_options_direct": ActionSpec(
        "Text",
        "options_only",
        "direct",
        "Generates from answer options without the original question.",
    ),
    "tc_mean": ActionSpec(
        "TextConsistency",
        "paraphrase",
        "mean",
        "Averages textual paraphrase evidence.",
    ),
    "b5_joint_strong_osmax": ActionSpec(
        "Ablation",
        "option_shuffle+strong_blur",
        "max",
        "Legacy strong joint strategy kept for ablation, not the clean main method.",
    ),
    "b5_joint_strong_os1": ActionSpec(
        "Joint",
        "option_shuffle_1+strong_blur",
        "direct",
        "Combines one option-shuffle counterfactual with strong-blur visual contrast.",
    ),
    "b5_joint_center_os1": ActionSpec(
        "Joint",
        "option_shuffle_1+center_mask",
        "direct",
        "Combines one option-shuffle counterfactual with center-mask visual contrast.",
    ),
    "tcg_both_strong_qmask_b2_t0.05": ActionSpec(
        "Ablation",
        "question_mask+strong_blur",
        "direct",
        "Single-text/single-visual open-ended ablation.",
    ),
    "tcg_both_center_qmask_b2_t0.05": ActionSpec(
        "Ablation",
        "question_mask+center_mask",
        "direct",
        "Single-text/single-visual open-ended ablation.",
    ),
    "floor_sci5_ours_aug_b0.1_g2_t0.3": ActionSpec(
        "Conservative",
        "2text+2visual",
        "sci5_like",
        "Legacy executable name for the conservative strategy.",
    ),
    OLD_CLEAN_FALLBACK: ActionSpec(
        "Conservative",
        "2text+2visual",
        "legacy_alias",
        "Legacy public id for the conservative strategy.",
    ),
    CLEAN_FALLBACK: ActionSpec(
        "Conservative",
        "2text+2visual",
        "conservative",
        "MCQ uses OS1+OS2; open-ended uses QuestionMask+OptionsOnly; both use StrongBlur+CenterMask.",
    ),
    "joint_strong_blur_option_shuffle1_b0.2_g2.5_t0.2": ActionSpec(
        "Joint",
        "option_shuffle_1+strong_blur",
        "direct",
        "Lightweight joint correction: one option-shuffle text probe plus one strong-blur visual probe.",
    ),
    "joint_center_mask_option_shuffle1_b0.2_g2.5_t0.2": ActionSpec(
        "Joint",
        "option_shuffle_1+center_mask",
        "direct",
        "Lightweight joint correction: one option-shuffle text probe plus one center-mask visual probe.",
    ),
    "joint_strong_blur_question_mask_b0.2_g2.5_t0.2": ActionSpec(
        "Joint",
        "question_mask+strong_blur",
        "direct",
        "Lightweight joint correction: one question-mask text probe plus one strong-blur visual probe.",
    ),
    "joint_center_mask_question_mask_b0.2_g2.5_t0.2": ActionSpec(
        "Joint",
        "question_mask+center_mask",
        "direct",
        "Lightweight joint correction: one question-mask text probe plus one center-mask visual probe.",
    ),
    SCI_BOTH: ActionSpec(
        "Ablation",
        "blank_noise+paraphrase",
        "max",
        "Published SCI-style conservative strategy kept for ablation, not the clean main method.",
    ),
    SCI_OS_BOTH: ActionSpec(
        "Ablation",
        "blank_noise+option_shuffle",
        "max",
        "SCI-style option-shuffle fusion kept for ablation, not the clean main method.",
    ),
}


def param_suffix(beta: float, gamma: float, theta: float) -> str:
    return f"b{beta:g}_g{gamma:g}_t{theta:g}"


def both_action(source: str, beta: float, gamma: float, theta: float) -> str:
    return f"both_max_{source}_{param_suffix(beta, gamma, theta)}"


def joint_os_action(source: str, beta: float, gamma: float, theta: float) -> str:
    return f"joint_os_{source}_{param_suffix(beta, gamma, theta)}"


def main_visual_action(source: str, strength: str) -> str:
    beta, gamma, theta = MAIN_VC_PARAMS[strength]
    return both_action(source, beta, gamma, theta)


def main_joint_action(text_probe: str, source: str, strength: str) -> str:
    beta, gamma, theta = MAIN_VC_PARAMS[strength]
    return f"joint_{source}_{text_probe}_{param_suffix(beta, gamma, theta)}"


def unique(actions: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for action in actions:
        if action not in seen:
            seen.add(action)
            out.append(action)
    return out


def main_method_slot_actions() -> dict[str, list[str]]:
    """Paper-facing clean main-method candidate set.

    This is the design contract: Text and Visual are lightweight routes, Joint
    composes one text probe with one visual probe, and Conservative is the
    conservative correction when routing evidence is weak or unreliable.

    Conservative is FORMAT-aware inside the executable action: MCQ uses
    OS1+OS2+StrongBlur+CenterMask; open-ended uses
    QuestionMask+OptionsOnly+StrongBlur+CenterMask. Published SCI/OOD-style full
    ablation remains an ablation action, not a main-method floor.
    """
    visual_actions = unique([
        main_visual_action(source, strength)
        for source in MAIN_VISUAL_SOURCES
        for strength in MAIN_VISUAL_STRENGTHS
    ])
    mcq_joint = [
        "joint_strong_blur_option_shuffle1_b0.2_g2.5_t0.2",
        "joint_center_mask_option_shuffle1_b0.2_g2.5_t0.2",
    ]
    oth_joint = [
        "joint_strong_blur_question_mask_b0.2_g2.5_t0.2",
        "joint_center_mask_question_mask_b0.2_g2.5_t0.2",
    ]
    fallback_mcq = [CLEAN_FALLBACK]
    fallback_oth = [CLEAN_FALLBACK]
    return {
        "floor:mcq": fallback_mcq,
        "floor:oth": fallback_oth,
        "tc:mcq": ["b5_tc_os1_direct"],
        "tc:oth": ["tcg_qmask_direct", "tcg_options_direct"],
        "tc_low:oth": ["tcg_options_direct", "tcg_qmask_direct"],
        "tc_high:oth": ["tcg_qmask_direct", "tcg_options_direct"],
        "vc:mcq": visual_actions,
        "vc:oth": visual_actions,
        "both:mcq": mcq_joint,
        "both:oth": oth_joint,
    }


def readable_action_name(action: str) -> str:
    spec = ACTION_SPECS.get(action)
    if spec is not None:
        return spec.readable_name
    if action.startswith("both_max_"):
        return "Visual:" + action.removeprefix("both_max_")
    if action.startswith("joint_os_"):
        return "Joint:" + action.removeprefix("joint_os_")
    if action.startswith("joint_"):
        return "Joint:" + action.removeprefix("joint_")
    return action


def public_action_name(action: str) -> str:
    """Paper-facing action name; executable ids stay unchanged for cache lookup."""
    if action in PUBLIC_ACTION_NAMES:
        return PUBLIC_ACTION_NAMES[action]
    if action.startswith("both_max_"):
        source = action.removeprefix("both_max_").split("_b", 1)[0]
        return "Visual-" + source.replace("_", " ").title().replace(" ", "")
    return readable_action_name(action)


def readable_slot_name(slot: str) -> str:
    return ROUTE_DISPLAY_NAMES.get(slot, slot)


def qwen_slot_actions() -> dict[str, list[str]]:
    """Legacy executable compact grammar instance used by prior Qwen routing.

    This preserves old cached experiments. For the paper-facing clean design,
    use main_method_slot_actions(), which removes SCI/OOD main candidates and
    uses the fixed 2TC+2VC conservative strategy.
    """
    return {
        "floor:mcq": ["b5_joint_strong_os1", "b5_joint_center_os1"],
        "floor:oth": ["tcg_both_strong_qmask_b2_t0.05", "tcg_both_center_qmask_b2_t0.05"],
        "tc:mcq": ["b5_tc_os1_direct"],
        "tc:oth": ["tcg_qmask_direct", "tcg_options_direct"],
        "tc_low:oth": ["tcg_options_direct", "tcg_qmask_direct"],
        "tc_high:oth": ["tcg_qmask_direct", "tcg_options_direct"],
        "vc:mcq": ["p1_core", "vc_ood_b0.1_t0.1"],
        "vc:oth": ["p1_core", "vc_ood_b0.1_t0.1"],
        "both:mcq": ["b5_joint_strong_os1", "b5_joint_center_os1", SCI_OS_BOTH, SCI_BOTH],
        "both:oth": [SCI_BOTH],
    }


def llava_slot_actions(
    both_betas: Iterable[float],
    both_gammas: Iterable[float],
    both_thetas: Iterable[float],
    joint_betas: Iterable[float],
    joint_gammas: Iterable[float],
    joint_thetas: Iterable[float],
    sources: Iterable[str] = SOURCE_VARIANTS,
) -> dict[str, list[str]]:
    """Legacy executable grammar instance used by prior LLaVA routing.

    This keeps OOD/SCI-compatible candidates needed to reproduce confirmed
    historical scores. It is not the clean main-method candidate contract.
    """
    both = [
        both_action("ood", beta, gamma, theta)
        for beta in both_betas
        for gamma in both_gammas
        for theta in both_thetas
    ]
    source = [
        both_action(source_name, beta, gamma, theta)
        for source_name in sources
        for beta in both_betas
        for gamma in both_gammas
        for theta in both_thetas
    ]
    joint_os = [
        joint_os_action(source_name, beta, gamma, theta)
        for source_name in sources
        for beta in joint_betas
        for gamma in joint_gammas
        for theta in joint_thetas
    ]
    fullgen_ood = both_action("ood", 0.5, 2.5, 0.2)
    open_ended_floor = [fullgen_ood] if fullgen_ood in both else both
    return {
        "floor:mcq": both,
        "floor:oth": open_ended_floor,
        "tc:mcq": ["b5_tc_os1_direct"],
        "tc_high:oth": ["tcg_qmask_direct", "tcg_options_direct"],
        "tc_low:oth": ["tcg_options_direct", "tcg_qmask_direct"],
        "vc:mcq": ["p1_core"] + source,
        "vc:oth": ["p1_core"] + source,
        "both:mcq": joint_os,
        "both:oth": open_ended_floor,
    }


def grammar_manifest(slot_actions: dict[str, list[str]]) -> dict:
    return {
        "route_names": ROUTE_DISPLAY_NAMES,
        "slot_actions": slot_actions,
        "readable_slot_actions": {
            f"{readable_slot_name(slot.split(':', 1)[0])}:{slot.split(':', 1)[1]}": [
                public_action_name(action) for action in actions
            ]
            for slot, actions in slot_actions.items()
        },
    }
