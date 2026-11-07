import matplotlib
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
# stop runtime errors and speed up plotting
matplotlib.use('Agg')
mplstyle.use('fast')
plt.minorticks_off()

import os
import numpy as np
import seaborn as sns
import torchaudio
from .plot_helper_sys import TITLE_FONT_SIZE, MAIN_LABEL_FONT_SIZE, TRANSCRIPTION_FONT_SIZE, DPI_AMOUNT, TITLE_PAD, SUPTITLE_PAD
from .plot_helper_sys import sq_ast_dim_str

# Individual plots

SPEC_TIME_HOP = 0.01
TICK_HOP = 0.5

def get_ticks(df_row, tick_hop, plot_hop):
    '''
    Helper function to get timestamps for audio segment in plots

    Parameters
    ----------
    df_row (pandas.dataframe) : Dataset row of information for segment
    tick_hop (float) : Hop to scale indices to
    plot_hop (float) : Time per sample in segment
    
    Returns
    ----------
    tick_times (numpy.array) : Time labels for the ticks extracted
    tick_indices (numpy.array) : Indices for the ticks extracted
    '''
    file_wav_segment_info = (df_row["wav_start"], df_row["wav_end"])
    file_fs = df_row["file_fs"]

    start_time = file_wav_segment_info[0] / file_fs
    first_tick_time = np.ceil(start_time / tick_hop) * tick_hop
    duration = (file_wav_segment_info[1] - file_wav_segment_info[0]) / file_fs
    tick_times = np.arange(first_tick_time, start_time + duration, tick_hop)
    tick_indices = np.array([(t - start_time) / plot_hop for t in tick_times])

    return tick_times, tick_indices


def plot_saliency_with_pdsm(
        config_file, df_row,
        spectrogram, saliency_map, fbank_length, 
        dim_index, all_dims, dim_phon, dim_pdsm, 
        output_dir
    ):
    '''
    Plots a figure of the Spectrogram with most important phonemes highlighted
    Underneath a Saliency Map is plotted to validate phonemes selected

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    df_row (pandas.dataframe) : Row of data gathered by analysis
    spectrogram (np.array) : SQ_AST dataset spectrogram for audio file
    saliency_map (np.array) : Extracted and scales SQ_AST saliency for audio file for given dimension
    fbank_length (int) : Time dimension of spectrogram (as max is parsed into spectrogram arg)
    dim_index (int) : Index of SQ_AST dimensions
    all_dims (list) : SQ_AST dimensions to run
    dim_phon (list) : List of important phonemes including their start and end times
    dim_pdsm (np.array) : Mask of important phonemes
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    dim = all_dims[dim_index]
    file_path = df_row["file_path"]
    file_segment = df_row["wav_segment"]
    sq_ast_pred_i = df_row[f"sq_{dim}"]

    tick_times, tick_indices = get_ticks(df_row, TICK_HOP, SPEC_TIME_HOP)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='gray')

    # RGBA image of pdsm
    pdsm_image = np.array([np.ceil(dim_pdsm), np.zeros_like(dim_pdsm), np.zeros_like(dim_pdsm), dim_pdsm])
    pdsm_image = np.moveaxis(pdsm_image, 0, -1)
    plt.imshow(pdsm_image, aspect='auto', origin='lower')
    
    plt.xlim(0, fbank_length)
    plt.xticks(tick_indices, tick_times, fontsize=MAIN_LABEL_FONT_SIZE)
    plt.yticks(fontsize=MAIN_LABEL_FONT_SIZE)
    plt.ylim(0, 128)

    y_offset_index = 0
    for phon in dim_phon:
        phon_text = str(y_offset_index+1) + "\n" + phon[1]
        plt.text((phon[2]+phon[3])*0.5, 0.88*128, phon_text, fontdict={"fontsize":TRANSCRIPTION_FONT_SIZE, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"})
        y_offset_index += 1
    plt.title("Mel-Spectrogram With Most Important Phonemes", fontsize=MAIN_LABEL_FONT_SIZE, pad=TITLE_PAD)
    plt.xlabel("Time in Seconds", fontdict={"fontsize":MAIN_LABEL_FONT_SIZE})
    plt.ylabel("Mel Frequency Bin", fontdict={"fontsize":MAIN_LABEL_FONT_SIZE})

    plt.subplot(2, 1, 2)
    plt.imshow(saliency_map, alpha=0.6, aspect='auto', origin='lower', cmap='jet')
    plt.xlim(0, fbank_length)
    plt.xticks(tick_indices, tick_times, fontsize=MAIN_LABEL_FONT_SIZE)
    plt.yticks(fontsize=MAIN_LABEL_FONT_SIZE)
    plt.ylim(0, 128)
    plt.title(f"Important Areas for {sq_ast_dim_str[dim]}", fontsize=MAIN_LABEL_FONT_SIZE, pad=TITLE_PAD)
    plt.xlabel("Time in Seconds", fontsize=MAIN_LABEL_FONT_SIZE)
    plt.ylabel("Mel Frequency Bin", fontsize=MAIN_LABEL_FONT_SIZE)
    
    if file_path == "":
        str_file_path = config_file["dataset_name"]
    else:
        str_file_path = file_path

    str_wav_segment = ""
    if df_row["total_wav_segments"] > 1:
        str_wav_segment = f"(segment: {file_segment}), "

    if config_file["pdsm"]["k_method"] == "threshold":
        plt.suptitle(
            f"File: {str_file_path}, {str_wav_segment}{sq_ast_dim_str[dim]}: {np.round(sq_ast_pred_i, 1):.1f}\n(With the {config_file["pdsm"]["k"]:.0f} Most Important Phonemes Highlighted)", 
            fontsize=TITLE_FONT_SIZE
        )
    elif config_file["pdsm"]["k_method"] == "percent":
        plt.suptitle(
            f"File: {str_file_path}, {str_wav_segment}{sq_ast_dim_str[dim]}: {np.round(sq_ast_pred_i, 1):.1f}\n(With the {100*config_file["pdsm"]["k"]:.0f}% Most Important Phonemes Highlighted)", 
            fontsize=TITLE_FONT_SIZE
        )
    elif config_file["pdsm"]["k_method"] == "adaptive":
        scale = config_file["pdsm"].get("adaptive_lambda", 1.0)
        plt.suptitle(
            f"File: {str_file_path}, {str_wav_segment}{sq_ast_dim_str[dim]}: {sq_ast_pred_i:.1f}\n"
            f"Adaptive PDSM: score > mean + {scale:g} * std; {len(dim_phon)} phonemes selected",
            fontsize=TITLE_FONT_SIZE,
        )

    plt.tight_layout(pad=SUPTITLE_PAD)
    plt.savefig(os.path.join(output_dir, f"{file_segment}_{dim}_Phoneme.png"), dpi=DPI_AMOUNT, bbox_inches='tight')
    plt.clf()
    plt.close()

    return 0


def plot_saliency_jointgrid(
        config_file, df_row,
        spectrogram, saliency_map, kde_x_est, kde_y_est, 
        dim_index, all_dims, result_word_alignment, 
        output_dir
    ):
    '''
    Plots a figure of the Spectrogram with a Saliency overlay and KDE along each axis for a singular audio file

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    df_row (pandas.dataframe) : Row of data gathered by analysis
    spectrogram (np.array) : SQ_AST dataset spectrogram for audio file
    saliency_map (np.array) : Extracted and scales SQ_AST saliency for audio file for given dimension
    kde_x_est (np.array) : KDE for the time dimension
    kde_y_est (np.array) : KDE for the frequency dimension
    all_dims (list) : SQ_AST dimensions to run
    dim_phon (list) : List of important phonemes including their start and end times
    result_word_alignment (list) : ASR word alignment for the given audio file
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    dim = all_dims[dim_index]
    file_path = df_row["file_path"]
    file_segment = df_row["wav_segment"]
    sq_ast_pred_i = df_row[f"sq_{dim}"]

    tick_times, tick_indices = get_ticks(df_row, TICK_HOP, SPEC_TIME_HOP)

    h, w = saliency_map.shape
    x_flat = np.arange(w)
    y_flat = np.arange(h)

    g = sns.JointGrid(height=4, ratio=5, space=0.1)
    g.fig.set_size_inches(15, 7)
    g.ax_joint.imshow(spectrogram.T, aspect='auto', cmap='gray', origin='lower')
    pos = g.ax_joint.imshow(saliency_map, aspect='auto', cmap='jet', origin='lower', alpha=0.25)

    g.ax_joint.set_xticks(tick_indices, tick_times)
    g.ax_joint.tick_params(axis='x', labelsize=MAIN_LABEL_FONT_SIZE)
    g.ax_joint.set_xlabel("Time in Seconds", fontsize=MAIN_LABEL_FONT_SIZE)

    g.ax_joint.set_ylabel("Mel Frequency Bins", fontsize=MAIN_LABEL_FONT_SIZE)
    g.ax_joint.tick_params(axis='y', labelsize=MAIN_LABEL_FONT_SIZE)

    
    last_end = 0
    binary_offset = 0
    for word_segment in result_word_alignment:
        if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
            if int(word_segment["start"]/SPEC_TIME_HOP) != last_end:
                g.ax_joint.plot(
                    [int(word_segment["start"]/SPEC_TIME_HOP), int(word_segment["start"]/SPEC_TIME_HOP)], 
                    [0, 128], 
                    c="black", 
                    alpha=0.5
                )
            g.ax_joint.plot(
                [int(word_segment["end"]/SPEC_TIME_HOP), int(word_segment["end"]/SPEC_TIME_HOP)], 
                [0, 128], 
                c="black", 
                alpha=0.5
            )
            last_end = int(word_segment["end"]/SPEC_TIME_HOP)
            g.ax_joint.text(
                (int(word_segment["start"]/SPEC_TIME_HOP) + int(word_segment["end"]/SPEC_TIME_HOP))*0.5, 
                128*(0.95-binary_offset*0.05), 
                word_segment["word"], 
                fontdict={"fontsize":TRANSCRIPTION_FONT_SIZE, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
            )
            binary_offset = not binary_offset

    g.ax_marg_x.plot(x_flat, kde_x_est)
    g.ax_marg_y.plot(kde_y_est, y_flat)

    g.ax_joint.set_xlim(0, w)
    g.ax_joint.set_ylim(0, h)

    if file_path == "":
        str_file_path = config_file["dataset_name"]
    else:
        str_file_path = file_path

    str_wav_segment = ""
    if df_row["total_wav_segments"] > 1:
        str_wav_segment = f"(segment: {file_segment}), "

    title = ""
    if config_file["pdsm"]["k_method"] == "threshold":
        title = f"File: {str_file_path}, {str_wav_segment}{sq_ast_dim_str[dim]}: {np.round(sq_ast_pred_i, 1):.1f}\nwith KDE for Time/Freq"
    elif config_file["pdsm"]["k_method"] in {"percent", "adaptive"}:
        title = f"File: {str_file_path} {str_wav_segment}{sq_ast_dim_str[dim]}: {np.round(sq_ast_pred_i, 1):.1f}\nwith KDE for Time/Freq"
    
    g.fig.suptitle(
        title, fontsize=TITLE_FONT_SIZE
    )

    plt.tight_layout(pad=SUPTITLE_PAD)
    plt.savefig(os.path.join(output_dir, f"{file_segment}_{dim}_KDE_Word.png"), dpi=DPI_AMOUNT, bbox_inches='tight')
    plt.clf()
    plt.close()

    return 0


def plot_kde_along_waveform(
        config_file, df_row,
        time_kdes, result_word_alignment, 
        all_dims, output_dir, 
    ):
    '''
    Plots a figure of ASR Word Confidence over a single audio file

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    df_row (pandas.dataframe) : Row of data gathered by analysis
    time_kdes (numpy.array) : Kernel density estimation over time for a single audio file
    result_word_alignment (list) : ASR word alignment for the given audio file
    all_dims (list) : SQ_AST dimensions to run
    sq_ast_scores (numpy.array) : Scores for SQ_AST output across all dimensions for audio file
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    file_path = df_row["file_path"]
    file_segment = df_row["wav_segment"]
    channel = df_row["wav_channel"]
    file_wav_segment_info = (df_row["wav_start"], df_row["wav_end"])
    sq_ast_scores = [float(df_row[f"sq_{dim}"]) for dim in all_dims]

    tick_times, tick_indices = get_ticks(df_row, TICK_HOP, 1.0/df_row["file_fs"])

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])
    audio_test, fs_test = torchaudio.load(os.path.join(input_dir, file_path))
    # get channel of waveform
    if audio_test.shape[0] > 1:
        audio_test = audio_test[channel, :]
    else:
        audio_test = audio_test.squeeze()
    # get segment of waveform
    audio_test = audio_test[file_wav_segment_info[0]:file_wav_segment_info[1]].numpy()

    audio_test = (audio_test.astype(np.float32))/np.max(np.abs(audio_test))
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.3, 1]}, figsize=(15, 5))
    
    for dim_index in range(len(all_dims)):
        dim = all_dims[dim_index]
        if sq_ast_scores[dim_index] <= config_file["score_threshold"]:
            time_kde_rescaled = np.interp(
                np.linspace(0, len(time_kdes[dim_index]), len(audio_test)), 
                np.linspace(0, len(time_kdes[dim_index]), len(time_kdes[dim_index])), 
                time_kdes[dim_index]
            )
            time_kde_rescaled = time_kde_rescaled/np.max(time_kde_rescaled)
            axs[0].plot(time_kde_rescaled, label=sq_ast_dim_str[dim])
    
    axs[0].set_xlim(0, len(time_kde_rescaled))
    axs[0].set_xticks([0], [None])
    axs[0].axis("off")
    axs[0].set_yticks([0], [None])
    axs[0].tick_params(axis='x', labelsize=MAIN_LABEL_FONT_SIZE)
    axs[0].tick_params(axis='y', labelsize=MAIN_LABEL_FONT_SIZE)
    axs[0].set_ylim(0, 1.05) # extra 5%

    last_end = 0
    for word_segment in result_word_alignment:
        if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
            if word_segment["start"]*fs_test != last_end:
                axs[0].plot(
                    [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                    [0, np.max(time_kde_rescaled)*1.05], 
                    c="black", 
                    alpha=0.5
                )
            axs[0].plot(
                [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
                [0, np.max(time_kde_rescaled)*1.05], 
                c="black", 
                alpha=0.5
            )
            last_end = word_segment["end"]*fs_test

    axs[0].legend(
        bbox_to_anchor=[0.0, 0.0], loc='lower left', fontsize=9
    )

    axs[1].plot(audio_test)

    last_end = 0
    for word_segment in result_word_alignment:
        if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
            if word_segment["start"]*fs_test != last_end:
                axs[1].plot(
                    [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                    [-1, +1], 
                    c="black", 
                    alpha=0.5
                )
            axs[1].plot(
                [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
                [-1, +1], 
                c="black", 
                alpha=0.5
            )
            last_end = word_segment["end"]*fs_test

            axs[1].text(
                (word_segment["start"]*fs_test+word_segment["end"]*fs_test)*0.5, 
                1.1, 
                word_segment["word"], 
                fontdict={"fontsize":TRANSCRIPTION_FONT_SIZE, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
            )

    axs[1].set_xlim(0, len(audio_test))
    axs[1].set_xticks(tick_indices, tick_times)
    axs[1].tick_params(axis='x', labelsize=MAIN_LABEL_FONT_SIZE)
    axs[1].set_xlabel("Time in seconds", fontsize=MAIN_LABEL_FONT_SIZE)

    axs[1].set_ylim(-1, 1)
    axs[1].set_yticks([0], [None])
    axs[1].tick_params(axis='y', labelsize=MAIN_LABEL_FONT_SIZE)
    axs[1].set_ylabel("Amplitude", fontsize=MAIN_LABEL_FONT_SIZE)

    if file_path == "":
        str_file_path = config_file["dataset_name"]
    else:
        str_file_path = file_path

    str_wav_segment = ""
    if df_row["total_wav_segments"] > 1:
        str_wav_segment = f"(segment: {file_segment}), "

    plt.suptitle(f"File: {str_file_path}, {str_wav_segment}Importance over Time\nFor Sound Quality Metrics", fontsize=TITLE_FONT_SIZE)

    plt.tight_layout(pad=SUPTITLE_PAD)
    plt.savefig(os.path.join(output_dir, f"{file_segment}_KDE_Time_Word.png"), dpi=DPI_AMOUNT, bbox_inches='tight')
    plt.clf()
    plt.close()

    return 0


def plot_asr_confidence_along_waveform(
        config_file, df_row, result_word_alignment, 
        output_dir
    ):
    '''
    Plots a figure of ASR Word Confidence over a single audio file

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    file_path (os.path) : Path to the wav file
    file_segment (int) : Segment index of audio file
    channel (int) : channel of wavfile to read
    segment_info (tuple) : start and end samples of segment
    file_idx (int) : Original dataset index of the wav file
    result_word_alignment (list) : ASR word alignment for the given audio file
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    file_path = df_row["file_path"]
    file_segment = df_row["wav_segment"]
    channel = df_row["wav_channel"]
    file_wav_segment_info = (df_row["wav_start"], df_row["wav_end"])

    tick_times, tick_indices = get_ticks(df_row, TICK_HOP, 1.0/df_row["file_fs"])

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])
    audio_test, fs_test = torchaudio.load(os.path.join(input_dir, file_path))
    # get channel of waveform
    if audio_test.shape[0] > 1:
        audio_test = audio_test[channel, :]
    else:
        audio_test = audio_test.squeeze()
    # get segment of waveform
    audio_test = audio_test[file_wav_segment_info[0]:file_wav_segment_info[1]].numpy()
    audio_test = (audio_test.astype(np.float32))/np.max(np.abs(audio_test))
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.3, 1]}, figsize=(15, 5))
    
    confidence = np.zeros(shape=(len(audio_test)), dtype=np.float32)

    last_end = 0
    for word_segment in result_word_alignment:

        if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
            confidence[int(word_segment["start"]*fs_test):int(word_segment["end"]*fs_test)] = np.float32(word_segment["score"])
            if word_segment["start"]*fs_test != last_end:
                axs[0].plot(
                    [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                    [0, 1.05], 
                    c="black", 
                    alpha=0.5
                )
            axs[0].plot(
                [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
                [0, 1.05], 
                c="black", 
                alpha=0.5
            )
            last_end = word_segment["end"]*fs_test

    axs[0].plot(confidence)
    
    axs[0].set_xlim(0, len(audio_test))
    axs[0].set_xticks([0], [None])
    axs[0].tick_params(axis="x", labelsize=MAIN_LABEL_FONT_SIZE)

    axs[0].set_yticks([0, 1.0], ["0%", "100%"])
    axs[0].tick_params(axis="y", labelsize=MAIN_LABEL_FONT_SIZE)
    axs[0].yaxis.tick_right()
    axs[0].set_ylabel("Confidence", fontsize=MAIN_LABEL_FONT_SIZE)
    axs[0].set_ylim(0, 1.05) # extra 5%
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)

    axs[1].plot(audio_test)

    last_end = 0
    for word_segment in result_word_alignment:
        if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
            if word_segment["start"]*fs_test != last_end:
                axs[1].plot(
                    [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                    [-1, +1], 
                    c="black", 
                    alpha=0.5
                )
            axs[1].plot(
                [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
                [-1, +1], 
                c="black", 
                alpha=0.5
            )
            last_end = word_segment["end"]*fs_test

            axs[1].text(
                (word_segment["start"]*fs_test+word_segment["end"]*fs_test)*0.5, 
                1.1, 
                word_segment["word"], 
                fontdict={"fontsize":TRANSCRIPTION_FONT_SIZE, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
            )

    axs[1].set_xlim(0, len(audio_test))
    axs[1].set_xticks(tick_indices, tick_times)
    axs[1].tick_params(axis='x', labelsize=MAIN_LABEL_FONT_SIZE)
    axs[1].set_xlabel("Time in seconds", fontsize=MAIN_LABEL_FONT_SIZE)

    axs[1].set_ylim(-1, 1)
    axs[1].set_ylabel("Amplitude", fontsize=MAIN_LABEL_FONT_SIZE)
    axs[1].set_yticks([0], [None])
    axs[1].tick_params(axis='y', labelsize=MAIN_LABEL_FONT_SIZE)

    if file_path == "":
        str_file_path = config_file["dataset_name"]
    else:
        str_file_path = file_path

    str_wav_segment = ""
    if df_row["total_wav_segments"] > 1:
        str_wav_segment = f"(segment: {file_segment}), "

    plt.suptitle(f"File: {str_file_path}, {str_wav_segment}\nASR Confidence For Each Word", fontsize=TITLE_FONT_SIZE)

    plt.tight_layout(pad=SUPTITLE_PAD)
    plt.savefig(os.path.join(output_dir, f"{file_segment}_ASR_Confidence.png"), dpi=DPI_AMOUNT, bbox_inches='tight')
    plt.clf()
    plt.close()

    return 0

