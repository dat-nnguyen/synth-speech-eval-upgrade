# Original code Copyright (c) 2025 Wafaa Wardah (MIT License pasted below)
# Modifications by Ben Heritage 2026 to extract saliency
# """
# MIT License

# Copyright (c) 2025 Wafaa Wardah

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# """

import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from transformers import ASTConfig, ASTModel, ASTFeatureExtractor, logging as hf_logging
import numpy as np
import logging
logger = logging.getLogger(__name__)


hf_logging.set_verbosity_error()  # Silence unnecessary warnings from huggingface
torch.multiprocessing.set_sharing_strategy('file_system')


REF_ALL_DIMS = ["mos", "noi", "dis", "col", "loud"]
# COL_IDX = {d: i for i, d in enumerate(ALL_DIMS)}
SALIENCY_INTERP_SIZE = (128, 1024)

MAX_AUDIO_LEN = 10
SQ_FRAME_SIZE = 0.025
SQ_HOP_SIZE = 0.01
SQ_MEL_FREQ = SALIENCY_INTERP_SIZE[0]

# Values from SQ_AST code
DB_MEAN = -10.25446422 # calculated from the validation datasets from July 2025
DB_STD = 4.205750774 # calculated from the validation datasets from July 2025


def _load_cached_pretrained(model_class, model_name, **kwargs):
    """Read cached metadata first; contact the Hub only for a missing cache."""
    try:
        return model_class.from_pretrained(model_name, local_files_only=True, **kwargs)
    except OSError:
        return model_class.from_pretrained(model_name, **kwargs)


class ASTXL(torch.nn.Module):
    """AST architecture; load a full SQ-AST checkpoint before inference."""

    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(ASTXL, self).__init__()
        try:
            config = _load_cached_pretrained(ASTConfig, pretrained_model, output_attentions=True)
            # Each SQ-AST checkpoint contains the complete AST state and is
            # loaded strictly below. Downloading the AudioSet weights first
            # would fetch another backbone that is immediately overwritten.
            self.ast = ASTModel(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AST from {pretrained_model}") from e

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, output_attentions: bool = False) -> torch.Tensor:
        if output_attentions:
            outputs = self.ast(features, output_attentions=True)
            hidden_state = outputs.pooler_output            
            attentions = outputs.attentions

            pred = self.fc(hidden_state).squeeze()
            
            return pred, attentions[0]
        else:
            hidden_state = self.ast(features, output_attentions=False).pooler_output
            pred = self.fc(hidden_state).squeeze()
            return pred
    

class ASTVal(Dataset):
    def __init__(self, df, data_dir, db_mean, db_std):
        self.df = df
        self.data_dir = data_dir
        self.feature_extractor = _load_cached_pretrained(ASTFeatureExtractor, ASTXL.PRETRAINED_MODEL)
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for our 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True
        self.feature_extractor.mean = db_mean
        self.feature_extractor.std = db_std

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        TARGET_SR = 48_000

        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])
        waveform, sample_rate = torchaudio.load(file_name)

        # get channel of waveform
        if waveform.shape[0] > 1:
            waveform = waveform[self.df['wav_channel'].iloc[index], :]
        else:
            waveform = waveform.squeeze()

        # get segment of waveform
        waveform = waveform[self.df['wav_start'].iloc[index]:self.df['wav_end'].iloc[index]]

        # resample before segment
        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SR
            )
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        return index, features


def get_expected_fbank_shape(waveform, sample_rate):
    """
    Returns expected shape of frequency bank mel_spectrogram

    Parameters
    ----------
    waveform (np.array) : Raw audio file after reading
    sample_rate (int) : Audio sample rate (should be 48k)
    
    Returns
    ----------
    dl (dataloader) : The dataloader needed by SQ_AST
    ds (ASTVal) : The SQ_AST dataset to be used for later analysis and spectrogram access
    fbank_lengths (np.array) : Lengths of spectrograms for each audio file in dataset
    audio_files_txt (list) : List of strings to file paths
    """

    num_mel_bins = SALIENCY_INTERP_SIZE[0]
    frame_length = int(sample_rate * SQ_FRAME_SIZE)  # 25 ms
    frame_step = int(sample_rate * SQ_HOP_SIZE)    # 10 ms
    num_frames = int(np.floor((waveform.shape[1] - frame_length) / frame_step)) + 1
    return (num_mel_bins, num_frames)


def prepare_dataloader(input_dir, input_df, bs, num_workers):
    """
    Gathers the data in the input directory ready for the SQ_AST forward pass

    Parameters
    ----------
    data_dir (os.path) : Dataset directory
    input_df (pandas.DataFrame) : Dataframe conatining all input wav information
    bs (int) : SQ_AST batch size
    num_workers (int) : SQ_AST num_workers arguement
    
    Returns
    ----------
    dl (dataloader) : The dataloader needed by SQ_AST
    ds (ASTVal) : The SQ_AST dataset to be used for later analysis and spectrogram access
    fbank_lengths (np.array) : Lengths of spectrograms for each audio file in dataset
    """

    fbank_lengths = [[i, 0] for i in range(len(input_df))]
    for index, row in input_df.iterrows():
        audio, sample_rate = torchaudio.load(os.path.join(input_dir, row["file_path"]))
        fbank_lengths[index] = [index, get_expected_fbank_shape(audio[:, row["wav_start"]:row["wav_end"]], sample_rate)[1]]

    logger.info(f"Running SQ_AST Inference on {len(input_df)} audio files.")

    ds = ASTVal(input_df, input_dir, DB_MEAN, DB_STD)
    dl = DataLoader(dataset=ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    fbank_lengths = np.array(fbank_lengths)

    return dl, ds, fbank_lengths


def get_scaled_saliency_map(attn_map, saliency_interp_method):
    """
    Interpolates and scales the input saliency map to the size of the input spectrogram of SQ_AST

    Parameters
    ----------
    attn_map (np.array) : Input saliency map to interpolate up to size
    saliency_interp_method (str) : extracted attention for each layer
    
    Returns
    ----------
    scaled_attn_map (np.array) : Interpolated saliency map to fit size of the input spectrogram to SQ_AST

    """
    attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
    scaled_attn_map = F.interpolate(attn_map, size=SALIENCY_INTERP_SIZE, mode=saliency_interp_method)
    return scaled_attn_map
    

def tensor_attention_flow(attn_maps):
    """
    Approximates attention flow (using only CLS on each layer for efficiency)

    Parameters
    ----------
    attn_maps (np.array) : extracted attention for each layer
    
    Returns
    ----------
    cls_flow (torch.Tensor) : Flattened tensor for attention flow on CLS (removed <CLS> and <DISTILL> tokens)

    """
    num_layers, seq_len, _ = attn_maps.shape

    I = torch.eye(seq_len).to(attn_maps.device)
    A_adj = 0.5 * attn_maps + 0.5 * I # add identity for residual connections
    
    cls_flow = A_adj[0, 0, :] # CLS attention only!
    for i in range(1, num_layers):
        cls_flow = torch.min(cls_flow, A_adj[i]).sum(dim=-1)
        cls_flow /= cls_flow.sum() # renormalise
    return cls_flow[2:] # remove <CLS> and <DISTILL>


def tensor_attention_rollout(attn_maps):
    """
    Extracts attention rollout

    Parameters
    ----------
    attn_maps (np.array) : extracted attention for each layer
    
    Returns
    ----------
    cls_rollout (torch.Tensor) : Flattened tensor for attention rollout on CLS (removed <CLS> and <DISTILL> tokens)

    """
    num_layers, seq_len, _ = attn_maps.shape

    I = torch.eye(seq_len).to(attn_maps.device)
    A_adj = 0.5 * attn_maps + 0.5 * I
    rollout = A_adj[0]

    for i in range(1, num_layers):
        rollout = torch.matmul(A_adj[i], rollout)

    # remove <CLS> and <DISTILL>
    return rollout[0, 2:]


def tensor_grad_cam(activations, gradients):
    '''
    Get GradCAM heatmap for important patches to decision on scoring.

    Paramaters
    ----------
    activations (torch.tensor) : Activations stored from the SQ_AST forward pass
    gradients (torch.tensor) : Gradients stored from the SQ_AST backward pass
    
    Returns
    ----------
    cam (torch.tensor) : The saliency from GradCAM for the last attention layer in SQ_AST
    '''
    weights = torch.mean(gradients[0], dim=1)
    cam = torch.matmul(activations[0], weights.unsqueeze(-1))
    cam = cam.squeeze(-1)
    cam = cam[:, 2:].squeeze()
    cam = F.relu(cam)
    return cam


LOG_PERCENTAGES = 20

def print_log_progress(current, total, last_logged, text):
    '''
    Prints the percentage in 20% chunks to stop clogging of sq_ast logs

    Parameters
    ----------
    current (int) : current batch number
    total (int) : total number of batches
    last_logged (int) : last logged percentage (stop repeats)
    text (str) : text to prepend the percentage

    Returns
    ----------
    last_logged (int) : Number updated if logged
    '''
    percent = 100*(current/total)
    if percent >= last_logged + LOG_PERCENTAGES:
        last_logged = percent
        logger.info(f"{text}: {int(percent)}%")
    return last_logged


def get_pred_attn(method, dims, dl, device, bs, threshold_value, num_inputs):
    """
    Evaluates the SQ_AST outputs for given input and dimensions, and returns the attention flow

    Parameters
    ----------
    method (str) : Method of "Flow" or "GradCAM"
    dims (list) : Dimensions to run in SQ_AST
    dl (dataloader) : Dataloader for the ASTVal input dataset
    device (str) : "cpu"/"cuda"
    bs (int) : Batch size to run on SQ_AST
    threshold_value (float) : User defined threshold to filter predictions below
    num_inputs (int) : Number of inputs in input dataframe

    Returns
    ----------
    predictions (numpy.array) : SQ_AST prediction outputs shape (files, dimensions)
    saliency (numpy.array) : SQ_AST extracted saliency(files, dimensions, transformer layers, input tokens)

    """
    
    saliency = torch.Tensor(np.zeros(shape=(num_inputs, len(dims), 12, 101)))
    predictions = torch.Tensor(np.zeros(shape=(num_inputs, len(dims))))

    for dim_index in range(len(dims)):
        dim = dims[dim_index]
            
        if (method == "Flow") or (method == "Raw") or (method == "Rollout"):
            with torch.no_grad():
                with torch.inference_mode():
                    model = ASTXL()
                    model.load_state_dict(torch.load(f"src/models/weights/{dim}.pth", map_location=torch.device(device), weights_only=True))
                    model.to(device)
                    model.eval()

                    last_logged = 0
                    for batch_index, (index, batch_features) in enumerate(dl):
                        last_logged = print_log_progress(batch_index, dl.__len__(), last_logged, dim)
                        batch_features = batch_features.float().to(device)
                        pred, attentions = model(batch_features, output_attentions=True)
                        attentions = attentions.cpu().detach()
                        if (bs == 1) or (pred.dim() == 0):
                            predictions[index, dim_index] = 4*pred + 1 # store prediction for this SQ dimension
                            if 4*pred + 1 <= threshold_value:                            
                                if method == "Flow":
                                    attention_ext = tensor_attention_flow(attentions[0].cpu().detach())
                                elif method == "Raw":
                                    attention_ext = attentions[0].cpu().detach()[-1, 0, 2:]
                                elif method == "Rollout":
                                    attention_ext = tensor_attention_rollout(attentions[0].cpu().detach())

                                saliency[index, dim_index, :, :] = attention_ext.reshape(12, 101)

                        else:
                            for i in index:
                                predictions[i, dim_index] = 4*pred[i-int(bs*batch_index)] + 1 # store prediction for this SQ dimension
                                if 4*pred[i-int(bs*batch_index)] + 1 <= threshold_value:
                                    if method == "Flow":
                                        attention_ext = tensor_attention_flow(attentions[i-int(bs*batch_index)].cpu().detach())
                                    elif method == "Raw":
                                        attention_ext = attentions[i-int(bs*batch_index)].cpu().detach()[-1, 0, 2:]
                                    elif method == "Rollout":
                                        attention_ext = tensor_attention_rollout(attentions[i-int(bs*batch_index)].cpu().detach())

                                    saliency[i, dim_index, :, :] = attention_ext.reshape(12, 101)
                                    
                    logger.info(f"{dim}: 100%")

        elif method == "GradCAM":
            model = ASTXL()
            model.load_state_dict(torch.load(f"src/models/weights/{dim}.pth", map_location=torch.device(device), weights_only=True))
            model.to(device)
            model.eval()

            activations = [None]
            gradients = [None]

            def save_activation(module, input, output):
                activations[0] = output

            def save_gradient(module, grad_input, grad_output):
                gradients[0] = grad_output[0]

            target_layer = model.ast.encoder.layer[-1] # get last transformer layer
            forward_handle = target_layer.register_forward_hook(save_activation)
            backward_handle = target_layer.register_full_backward_hook(save_gradient)

            last_logged = 0
            for batch_index, (index, batch_features) in enumerate(dl):
                last_logged = print_log_progress(batch_index, dl.__len__(), last_logged, dim)
                batch_features = batch_features.float().to(device)
                # Scores are >= 1, so a threshold below 1 cannot request
                # saliency. Avoid retaining a gradient graph in score-only runs.
                need_grad = threshold_value >= 1
                batch_features.requires_grad = need_grad
                with torch.set_grad_enabled(need_grad):
                    pred_sal = model(batch_features, output_attentions=False)
                pred = pred_sal.cpu().detach()

                # batch_size always 1 for GradCAM
                predictions[index, dim_index] = 4*pred + 1 # store prediction for this SQ dimension
                if 4*pred + 1 <= threshold_value:
                    model.zero_grad()
                    pred_sal.backward()
                    grad_cam_saliency = tensor_grad_cam(activations, gradients).cpu().detach()
                    saliency[index, dim_index, :, :] = grad_cam_saliency.reshape(12, 101)
            logger.info(f"{dim}: 100%")

            # free handle memory
            forward_handle.remove()
            backward_handle.remove()

        else:
            raise ValueError("Saliency Method Not Given")
    

    return predictions, saliency


def sq_ast_validate_dims(sq_ast_dims):
    """
    Filters all dims given in config file, and returns a filtered list to stop any further errors

    Parameters
    ----------
    sq_ast_dims (list) : List of SQ_AST dimensions given in the config file

    Returns
    ----------
    refined_dims (list) : A refined list allowing only dimension that are valid
    """

    refined_dims = []
    for dim in sq_ast_dims:
        if dim in REF_ALL_DIMS:
            refined_dims.append(dim)
    if len(refined_dims) == 0:
        raise ValueError(f"Has to give at least one valid sq_ast dimension in config file: {REF_ALL_DIMS}")
    return refined_dims


def sq_ast_fw(config_file, input_df):
    """
    Completes the SQ_AST forward pass for whole dataset and gathers saliency map below threshold defined in config_file

    Parameters
    ----------
    config_file (dict) :  Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    input_df (pandas.DataFrame) : Dataframe conatining all input nformation

    Returns
    ----------
    ds (ASTVal) : this is the dataset that contains the preprocessing steps for SQ_AST
    predictions (np.array) : SQ_AST output predictions in the shape (files, dimension)
    saliency_map (np.array) : SQ_AST saliency output in the shape (files, dimensions, transformer layers, input tokens)
    fbank_lengths (np.array) : Time-dimension lengths of the input spectrograms (used in later plots)
    output_ind_df (pandas.dataframe) : Output dataframe for predictions
    """

    current_time = datetime.now()
    dims = config_file["sq_ast_dims"]
    bs = int(config_file["sq_ast"]["batch_size"])
    num_workers = int(0)

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])
    
    # get device from config and check if GPU is available
    device = config_file["device"]

    logger.info(f"Starting SQ_AST Preprocessing")

    if bs > len(input_df):
        bs = int(len(input_df))

    # Method only accepts batch size of 1
    if config_file["saliency"]["saliency_method"] == "GradCAM":
        bs = 1

    # create dataset for sq_ast
    dl, ds, fbank_lengths = prepare_dataloader(input_dir, input_df, bs, num_workers)

    # get the predictions and attention flows for each dimension
    predictions, saliency_map = get_pred_attn(config_file["saliency"]["saliency_method"], dims, dl, device, bs, config_file["score_threshold"], len(input_df))

    logger.info(f"SQ_AST inference completed in time: {datetime.now() - current_time}")

    # develop dataframe
    output_ind_df = input_df

    for index, dim in enumerate(dims):
        input_df[f"sq_{dim}"] = predictions.cpu().numpy()[:, index]

    predictions = predictions.cpu().numpy()
    saliency_map = saliency_map.cpu().numpy()

    return (
        ds, predictions, saliency_map, fbank_lengths, output_ind_df
    )


def get_thresholded_df(config_file, input_df):
    """
    Gets the filtered Dataframe with a given score threshold
    Returns the dataframe with all data lying below this threshold

    Parameters
    ----------
    config_file (dict) :  Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    input_df (pandas.dataframe) : Dataframe of SQ_AST outputs
    
    Returns
    ----------
    output_df_thresh (pandas.dataframe) : Thresholded dataframe

    """

    score_threshold = config_file["score_threshold"]
    dims = config_file["sq_ast_dims"]

    # get thresholded dataframe under the score threshold
    cols = [f"sq_{d}" for d in dims]
    output_df_thresh = input_df[(input_df[cols] <= score_threshold).any(axis=1)]
    
    output_df_thresh["index"] = np.arange(len(output_df_thresh))
    output_df_thresh = output_df_thresh.reset_index(names=["pre_threshold_index"])
    return output_df_thresh

