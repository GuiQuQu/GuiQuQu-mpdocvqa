"""
    计算anls指标
"""
from typing import List
import Levenshtein


def anls(predict_answer: List[str], ground_truth: List[List[str]]) -> float:
    """
    n = len(predict_answer),问题的数量
    predict_answer: List[str], 每个问题的预测答案
    ground_truth: List[List[str]], 每个问题的真实答案[一个问题可能存在多个]

    reference:
    1.https://rrc.cvc.uab.es/?ch=11&com=tasks
    2.https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
    """
    res = 0.0
    for pa, gts in zip(predict_answer, ground_truth):
        res += max([similarity(gt, pa) for gt in gts])
    return res / float(len(predict_answer))


# Normalized Levenshtein distance
def similarity(answer_ij: str, predict_i: str, tao: float = 0.5) -> float:
    maxlen = max(len(answer_ij), len(predict_i))
    edit_dist = Levenshtein.distance(answer_ij, predict_i)
    normalized_edit_dist = float(maxlen - edit_dist) / float(maxlen)
    if normalized_edit_dist >= tao:
        return normalized_edit_dist
    else:
        return 0.0
