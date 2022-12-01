from cdifflib import CSequenceMatcher

from nemo.utils import logging

logging.setLevel("INFO")

MATCH = "match"
NONMATCH = "non-match"
SEMIOTIC_TAG = "[SEMIOTIC_SPAN]"


def _get_alignment(a, b):
    """

	TODO: add

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
	"""
    a = a.lower().split()
    b = b.lower().split()

    s = CSequenceMatcher(None, a, b, autojunk=False)
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    s = s.get_matching_blocks()

    diffs = {}
    non_match_start_l = 0
    non_match_start_r = 0
    for match in s:
        l_start, r_start, length = match
        if non_match_start_l < l_start:
            while non_match_start_l < l_start:
                diffs[non_match_start_l] = (non_match_start_r, r_start, NONMATCH)
                non_match_start_l += 1

        for len_ in range(length):
            diffs[l_start + len_] = (r_start + len_, r_start + 1 + len_, MATCH)
        non_match_start_l = l_start + length
        non_match_start_r = r_start + length
    return diffs


def get_semiotic_spans(a, b):
    """returns list of different substrings between and b

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
	"""

    b = b.lower().split()
    s = CSequenceMatcher(None, a.lower().split(), b, autojunk=False)

    # do it here to preserve casing
    a = a.split()

    result = []
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    a_start_non_match, b_start_non_match = 0, 0
    # get not matching blocks
    for match in s.get_matching_blocks():
        a_start_match, b_start_match, match_len = match
        # we're widening the semiotic span to include 1 context word from each side, so not considering 1-word match here
        if match_len > 1:
            if a_start_non_match < a_start_match:
                result.append([[a_start_non_match, a_start_match], [b_start_non_match, b_start_match]])
            a_start_non_match = a_start_match + match_len
            b_start_non_match = b_start_match + match_len

    if a_start_non_match < len(a):
        result.append([[a_start_non_match, len(a)], [b_start_non_match, len(b)]])

    # add context (1 word from both sides)
    result_with_context = []
    for item in result:
        try:
            start_a, end_a = item[0]
            start_b, end_b = item[1]
        except:
            import pdb

            pdb.set_trace()
        if start_a - 1 >= 0 and start_b - 1 >= 0:
            start_a -= 1
            start_b -= 1
        if end_a + 1 <= len(a) and end_b + 1 <= len(b):
            end_a += 1
            end_b += 1
        result_with_context.append([[start_a, end_a], [start_b, end_b]])

    result = result_with_context
    semiotic_spans = []
    default_normalization = []
    a_masked = []
    start_idx = 0
    masked_idx = []
    for item in result:
        cur_start, cur_end = item[0]
        if start_idx < cur_start:
            a_masked.extend(a[start_idx:cur_start])
        start_idx = cur_end
        a_masked.append(SEMIOTIC_TAG)
        masked_idx.append(len(a_masked) - 1)

    if start_idx < len(a):
        a_masked.extend(a[start_idx:])

    for item in result:
        a_diff = ' '.join(a[item[0][0] : item[0][1]])
        b_diff = ' '.join(b[item[1][0] : item[1][1]])
        semiotic_spans.append(a_diff)
        default_normalization.append(b_diff)
        logging.debug(f"a: {a_diff}")
        logging.debug(f"b: {b_diff}")
        logging.debug("=" * 20)
    return result, semiotic_spans, a_masked, masked_idx, default_normalization


def _print_alignment(diffs, l, r):
    l = l.split()
    r = r.split()
    for l_idx, item in diffs.items():
        start, end, match_state = item
        print(f"{l[l_idx]} -- {r[start: end]} -- {match_state}")


def _adjust_span(semiotic_spans, norm_pred_diffs, pred_norm_diffs, norm_raw_diffs, raw: str, pred_text: str):
    """

	:param semiotic_spans:
	:param norm_pred_diffs:
	:param pred_norm_diffs:
	:param norm_raw_diffs:
	:param raw:
	:param pred_text:
	:return:
	"""
    standard_start = 0
    raw_text_list = raw.split()
    pred_text_list = pred_text.split()

    text_for_audio_based = {"semiotic": [], "standard": "", "pred_text": []}

    for idx, (raw_span, norm_span) in enumerate(semiotic_spans):

        raw_start, raw_end = raw_span
        raw_end -= 1

        # get the start of the span
        pred_text_start, _, status_norm_pred = norm_pred_diffs[norm_span[0]]
        new_norm_start = pred_norm_diffs[pred_text_start][0]
        raw_text_start = norm_raw_diffs[new_norm_start][0]

        # get the end of the span
        _, pred_text_end, status_norm_pred = norm_pred_diffs[norm_span[1] - 1]

        new_norm_end = pred_norm_diffs[pred_text_end - 1][1]
        raw_text_end = norm_raw_diffs[new_norm_end - 1][1]

        cur_semiotic_span = f"{' '.join(raw_text_list[raw_text_start:raw_text_end])}"
        cur_pred_text = f"{' '.join(pred_text_list[pred_text_start:pred_text_end])}"

        if len(cur_semiotic_span) > 0 and len(cur_pred_text) > 0:
            text_for_audio_based["semiotic"].append(cur_semiotic_span)
            text_for_audio_based["pred_text"].append(cur_pred_text)

        if standard_start < raw_text_start:
            text_for_audio_based["standard"] += " " + " ".join(raw_text_list[standard_start:raw_text_start])

        text_for_audio_based["standard"] += f" {SEMIOTIC_TAG} "
        standard_start = raw_text_end

    if standard_start < len(raw_text_list):
        text_for_audio_based["standard"] += ' '.join(raw_text_list[standard_start:])

    text_for_audio_based["standard"] = text_for_audio_based["standard"].replace("  ", " ").strip()
    return text_for_audio_based


def get_alignment(raw, norm, pred_text, verbose: bool = False):
    import time

    # start_time = time.time()
    semiotic_spans = get_semiotic_spans(raw, norm)[0]
    # print(f'Alignment 1: {round((time.time() - start_time) / 60, 2)} min.')

    # start_time = time.time()
    norm_raw_diffs = _get_alignment(norm, raw)
    # print(f'Alignment 2: {round((time.time() - start_time) / 60, 2)} min.')
    norm_pred_diffs = _get_alignment(norm, pred_text)
    pred_norm_diffs = _get_alignment(pred_text, norm)

    text_for_audio_based = _adjust_span(
        semiotic_spans, norm_pred_diffs, pred_norm_diffs, norm_raw_diffs, raw, pred_text
    )

    if verbose:
        print("=" * 40)
        for sem, pred in zip(text_for_audio_based["semiotic"], text_for_audio_based["pred_text"]):
            print(f"{sem} -- {pred}")
        print("=" * 40)
    return text_for_audio_based


if __name__ == "__main__":
    raw = "This, example: number 1,500 can be a very long one!, and can fail to produce valid normalization for such an easy number like 10,125 or dollar value $5349.01, and can fail to terminate, and can fail to terminate, and can fail to terminate, 452."
    norm = "This, example: number one thousand five hundred can be a very long one!, and can fail to produce valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and forty nine dollars and one cent, and can fail to terminate, and can fail to terminate, and can fail to terminate, four fifty two."
    pred_text = "this w example nuber viteen hundred can be a very h lowne one and can fail to produce a valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and fortyn nine dollars and one cent and can fail to terminate and can fail to terminate and can fail to terminate four fifty two"

    import json
    from time import perf_counter

    # det_manifest = f"/mnt/sdb/DATA/SPGI/normalization//sample_stt_en_conformer_ctc_large_default_HOUR.json"  # deter TN predictions stored in "pred_text" field
    # with open(det_manifest, "r") as f:
    #     for line in f:
    #         line = json.loads(line)
    #         norm = line["normalized"]
    #         raw = line["text"]
    #         pred_text = line["pred_text"]

    start_time = perf_counter()
    text_for_audio_based = get_alignment(raw, norm, pred_text, verbose=True)
    print(f'Execution time: {round((perf_counter() - start_time) / 60, 2)} min.')
    print(text_for_audio_based)

    # raw = "We have spent the last several years reshaping our branch network, upgrading technology and deepening our focus on our core 6 markets,"
    # norm = "we have spent the last several years reshaping our branch network upgrading technology and deepening our focus on our core six markets,"
    #
    # result = get_semiotic_spans(raw, norm)
