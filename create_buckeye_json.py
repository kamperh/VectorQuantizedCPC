#!/usr/bin/env python

"""
Create the json files for the Buckeye dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from pathlib import Path
from tqdm import tqdm
import json

forced_alignments_fn = Path(
    "/home/kamperh/endgame/projects/stellenbosch/bucktsong_awe_py3/data/buckeye_english.wrd"
    )
data_path = Path("/home/kamperh/endgame/datasets/buckeye")

speakers_dict = {
    "train": ["s02", "s04", "s05", "s08", "s12", "s16", "s03", "s06", "s10",
    "s11", "s13", "s38"],
    "val": ["s18", "s17", "s37", "s39", "s19", "s22", "s40", "s34"],
    "test": ["s20", "s25", "s27", "s01", "s26", "s31", "s29", "s23", "s24",
    "s32", "s33", "s30"]
    }


def uttlabel_to_uttkey(utterance):
    if utterance.startswith("nchlt"):
        # Xitsonga
        utt_split = utterance.split("_")
        speaker = utt_split.pop(2)
        utt_key = speaker + "_" + "-".join(utt_split)
    else:
        # Buckeye
        utt_key = utterance[0:3] + "_" + utterance[3:]
    return utt_key


def read_vad_from_fa(fa_fn, frame_indices=True):
    """
    Read voice activity detected (VAD) regions from a forced alignment file.

    The dictionary has utterance labels as keys and as values the speech
    regions as lists of tuples of (start, end) frame, with the end excluded.
    """
    vad_dict = {}
    prev_utterance = ""
    prev_token_label = ""
    prev_end_time = -1
    start_time = -1
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start_token, end_token, token_label = line.strip(
                ).split()
            start_token = float(start_token)
            end_token = float(end_token)
            # utterance = utterance.replace("_", "-")
            utt_key = uttlabel_to_uttkey(utterance)
            # utt_key = utterance[0:3] + "_" + utterance[3:]
            if utt_key not in vad_dict:
                vad_dict[utt_key] = []

            if token_label in ["SIL", "SPN"]:
                continue
            if prev_end_time != start_token or prev_utterance != utterance:
                if prev_end_time != -1:
                    utt_key = uttlabel_to_uttkey(prev_utterance)
                    # utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
                    if frame_indices:
                        # Convert time to frames
                        start = int(round(start_time * 100))
                        end = int(round(prev_end_time * 100)) + 1
                        vad_dict[utt_key].append((start, end))
                    else:
                        vad_dict[utt_key].append(
                            (start_time, prev_end_time)
                            )
                start_time = start_token

            prev_end_time = end_token
            prev_token_label = token_label
            prev_utterance = utterance

        utt_key = uttlabel_to_uttkey(prev_utterance)
        # utt_key = prev_utterance[0:3] + "_" + prev_utterance[3:]
        if frame_indices:
            # Convert time to frames
            start = int(round(start_time * 100))
            end = int(round(prev_end_time * 100)) + 1  # end index excluded
            vad_dict[utt_key].append((start, end))
        else:
            vad_dict[utt_key].append((start_time, prev_end_time))        
    return vad_dict


def main():

    # Read voice activity regions
    print("Reading:", forced_alignments_fn)
    vad_dict = read_vad_from_fa(forced_alignments_fn, frame_indices=False)

    # List of all segments
    print("Constructing metadata:")
    segments = []  # each entry is [in_path, offset, duration, out_path]
    for utt_key in tqdm(sorted(vad_dict)):
        speaker = utt_key[:3]
        utterance = utt_key[-3:]
        in_path = Path(speaker)/Path(speaker + utterance)
        wav_fn = (data_path/in_path).with_suffix(".wav")
        if not wav_fn.exists():
            print("Warning: Missing audio: {}".format(wav_fn))
            continue
        for (start, end) in vad_dict[utt_key]:
            start_frame = int(round(start * 100))
            end_frame = int(round(end * 100)) + 1
            segment_key = utt_key + "_{:06d}-{:06d}".format(
                start_frame, end_frame
                )
            offset = start
            duration = end - start
            out_path = Path(speaker)/segment_key
            segments.append([str(in_path), offset, duration, str(out_path)])
    print("Total no. segments: {}".format(len(segments)))

    # Extract segments for subsets
    split_segments = {"train": [], "val": [], "test": []}
    print("Splitting into train, val, test:")
    for segment in tqdm(segments):
        speaker = Path(segment[0]).parts[0]
        for split in speakers_dict:
            if speaker in speakers_dict[split]:
                segment[-1] = str(Path(split)/segment[-1])
                split_segments[split].append(segment)
                break

    # Write audio json files
    dataset_path = Path("datasets/buckeye")
    dataset_path.mkdir(parents=True, exist_ok=True)
    for split in sorted(split_segments):
        json_fn = (dataset_path/split).with_suffix(".json")
        print("Writing: {}".format(json_fn))
        with open(json_fn, "w") as f:
            json.dump(split_segments[split], f, indent=4)

    # Write speakers json file
    speakers = []
    for split in speakers_dict:
        speakers += speakers_dict[split]
    speakers = sorted(speakers)
    json_fn = dataset_path/"speakers.json"
    print("Writing: {}".format(json_fn))
    with open(json_fn, "w") as f:
        json.dump(speakers, f, indent=4)    


if __name__ == "__main__":
    main()
