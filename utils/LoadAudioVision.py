# -*- coding: utf-8 -*-
import lmdb
import tqdm
import math
import random
import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import io
import av

def load_audiovision_lmdb(txn, video_key):
    # (T, C, H, W) [0, 255]
    # In MOSI and MOSEI, each utterance video should be taken as a clip (same sentiment)
    # print(video_key)
    io_stream = io.BytesIO(txn.get(video_key.encode("utf-8")))
    audio_data, video_data = extract_audio_vision_from_video_binary(
        io_stream,
        multi_thread_decode=True,
    )
    # closing reading event and database environment
    # txn.abort()
    return audio_data, video_data

def extract_audio_vision_from_video_binary(
        in_mem_bytes_io, multi_thread_decode=False
    ):
    """
    Args:
        in_mem_bytes_io: binary from read file object
        # >>> with open(video_path, "rb") as f:
        # >>>     input_bytes = f.read()
        # >>> frames = extract_audio_vision_from_video_binary(input_bytes)
        # OR from saved binary in lmdb database
        # >>> env = lmdb.open("lmdb_dir", readonly=True)
        # >>> txn = env.begin()
        # >>> stream = io.BytesIO(txn.get(str("key").encode("utf-8")))
        # >>> frames = extract_audio_vision_from_video_binary(stream)
        # >>> from torchvision.utils import save_image
        # >>> save_image(frames[0], "path/to/example.jpg")  # save the extracted frames.
        target_fps: int, the input video may have different fps, convert it to
            the target video fps before frame sampling.
        multi_thread_decode: bool, if True, perform multi-thread decoding.

    Returns:
        torch.uint8, (T, C, H, W)
    """
    try:
        # Add `metadata_errors="ignore"` to ignore metadata decoding error.
        # When verified visually, it does not seem to affect the extracted frames.
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    except Exception as e:
        print(f"Exception in loading video binary: {e}")
        return None, None

    if multi_thread_decode:
        # Enable multiple threads for decoding.
        video_container.streams.audio[0].thread_type = "AUTO"
        video_container.streams.video[0].thread_type = "AUTO"

    # decoding and preprocessing audio data
    audio_frames = extract_transform_audio(video_container)

    # decoding and preprocessing video data
    video_frames = extract_transform_video(video_container)

    return audio_frames, video_frames


def extract_transform_audio(
    video_container,
    frame_length=25,
    num_mel_bins=128,
    target_length=204,
    target_sampling_rate=16000,
    clip_duration=1, # 2
    # clips_per_video=30, # 3
    mean=-4.268,
    std=9.138,
):
    # clip_duration, clips_per_video, frame_length set same as ImageBind
    clips_per_video = 30
    # decoding audio and resampling at different sample rate
    audio_frames_length = video_container.streams.audio[0].frames
    audio_rate = video_container.streams.audio[0].rate
    audio_length = video_container.streams.audio[0].duration * video_container.streams.audio[0].time_base
    clips_per_video = math.ceil(audio_length*2) # clip every 0.5 s

    # duration = video_container.streams.audio[0].duration
    frames, max_pts = pyav_decode_audio_stream(
        video_container,
        0,
        math.inf,
        video_container.streams.audio[0],
        {"audio": 0},
        target_sampling_rate=target_sampling_rate
    )
    frames = [torch.tensor(frame.to_ndarray()) for frame in frames]
    frames = torch.cat(frames, dim=-1)

    # frame_length=int((frames_length / fps) / audio_frames_length * 1000) # compute time-length of audio_frame

    # clip every 2 / 0.5 seconds for melspec
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, frames.size(1) / target_sampling_rate
    )
    audio_outputs = []
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = frames[
                        :,
                        int(clip_timepoints[0] * target_sampling_rate): int(
                            clip_timepoints[1] * target_sampling_rate
                        ),
                        ]
        waveform_melspec = waveform2melspec(
            waveform_clip,
            frame_length=frame_length,
            sample_rate=target_sampling_rate, num_mel_bins=num_mel_bins, target_length=target_length
        )  # setting same as ImageBind
        all_clips.append(waveform_melspec)

    # audio normalization
    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)
    # return audio_outputs

def extract_transform_video(
    video_container,
    target_fps=2, # micro expression duration < 1/2s
    sample_clip_idx=0,
    sample_num_clips=1,
    clip_duration=1, # 2
    # clips_per_video=50, # 5
):
    # (T, H, W, C), channels are RGB
    fps = float(video_container.streams.video[0].average_rate)
    frames_length = video_container.streams.video[0].frames
    duration = video_container.streams.video[0].duration

    video_length = video_container.streams.video[0].duration * video_container.streams.video[0].time_base
    clips_per_video = math.ceil(video_length*2) # clip every 0.5 s

    # decode the whole video to get the last frame pts
    frames, max_pts = pyav_decode_video_stream(
        video_container,
        0,
        math.inf,
        video_container.streams.video[0],
        {"video": 0},
    )

    # ImageBind video sampling
    frames = [frame.to_rgb().to_ndarray() for frame in frames]
    frames = torch.as_tensor(np.stack(frames))
    # (T, H, W, C) -> (C, T, H, W)
    if frames is not None:
        frames = frames.permute(3, 0, 1, 2)
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = UniformTemporalSubsample(num_samples=clip_duration*2) # 1 frame every 0.5s
    all_clips_timepoints = get_clip_timepoints(clip_sampler, frames.size(1) / fps)

    all_video = []
    for clip_timepoints in all_clips_timepoints:
        # Read the clip, get frames
        clip = frames[
            :,
            int(clip_timepoints[0] * fps) : int(clip_timepoints[1] * fps),
            :,
            :
        ]

        if clip is None:
            raise ValueError("No clip found")
        video_clip = frame_sampler(clip)
        video_clip = video_clip / 255.0  # since this is float, need 0-1

        all_video.append(video_clip)

    # Various length video and sample at fixed fps
    # clip_size = frames_length # the sampled clip will be the entire video
    # start_idx, end_idx = get_start_end_idx(
    #     frames_length,
    #     clip_size,
    #     sample_clip_idx,
    #     sample_num_clips,
    # )
    # # Perform temporal sampling from the decoded video.
    # num_frames = math.ceil(frames_length / fps * target_fps + 1) # round upper + 1 (start frame)
    # frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    # frames = [frame.to_rgb().to_ndarray() for frame in frames]
    # video_frames = torch.as_tensor(np.stack(frames))
    #
    # # (T, H, W, C) -> (T, C, H, W)
    # if video_frames is not None:
    #     video_frames = video_frames.permute(0, 3, 1, 2)
    # video_max_pts = max_pts

    # return video_frames

    # video normalization
    video_outputs = []
    video_transform = transforms.Compose(
        [
            ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    all_video = [video_transform(clip) for clip in all_video]
    all_video = SpatialCrop(224, num_crops=3)(all_video)

    all_video = torch.stack(all_video, dim=0)
    video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0)
    # return video_outputs

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (list(av.video.frame.VideoFrame)): a list of decoded video frames
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, len(frames) - 1).long().tolist()
    # frames = torch.index_select(frames, 0, index)
    frames = [frames[idx] for idx in index]
    return frames

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
            i.e., #frames to get at the original frame rate.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

def pyav_decode_video_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

def waveform2melspec(waveform, frame_length, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    # fbank better than MFCC, more similar as human ear
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=frame_length,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        print(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank

class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (Union[float, Fraction]): clip start time.
        clip_end_sec (Union[float, Fraction]): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: Union[float, Fraction]
    clip_end_sec: Union[float, Fraction]
    clip_index: int
    aug_index: int
    is_last_clip: bool

class ClipSampler(ABC):
    """
    Interface for clip samplers that take a video time, previous sampled clip time,
    and returns a named-tuple ``ClipInfo``.
    """

    def __init__(self, clip_duration: Union[float, Fraction]) -> None:
        self._clip_duration = Fraction(clip_duration)
        self._current_clip_index = 0
        self._current_aug_index = 0

    @abstractmethod
    def __call__(
        self,
        last_clip_end_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        pass

    def reset(self) -> None:
        """Resets any video-specific attributes in preperation for next video"""
        pass

class ConstantClipsPerVideoSampler(ClipSampler):
    """
    Evenly splits the video into clips_per_video increments and samples clips of size
    clip_duration at these increments.
    """

    def __init__(
        self, clip_duration: float, clips_per_video: int, augs_per_clip: int = 1
    ) -> None:
        super().__init__(clip_duration)
        self._clips_per_video = clips_per_video
        self._augs_per_clip = augs_per_clip

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): Not used for ConstantClipsPerVideoSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled.
            annotation (Dict): Not used by this sampler.
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.

        """
        max_possible_clip_start = Fraction(max(video_duration - self._clip_duration, 0))
        uniform_clip = Fraction(
            max_possible_clip_start, max(self._clips_per_video - 1, 1)
        )
        clip_start_sec = uniform_clip * self._current_clip_index
        clip_index = self._current_clip_index
        aug_index = self._current_aug_index

        self._current_aug_index += 1
        if self._current_aug_index >= self._augs_per_clip:
            self._current_clip_index += 1
            self._current_aug_index = 0

        # Last clip is True if sampled self._clips_per_video or if end of video is reached.
        is_last_clip = False
        if (
            self._current_clip_index >= self._clips_per_video
            or uniform_clip * self._current_clip_index > max_possible_clip_start
        ):
            self._current_clip_index = 0
            is_last_clip = True

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )

    def reset(self):
        self._current_clip_index = 0
        self._current_aug_index = 0

def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints

def pyav_decode_audio_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0, target_sampling_rate=-1,
):
    if target_sampling_rate==-1:
        resampler = None
    else:
        resampler = av.AudioResampler(
            format=container.streams.audio[0].format.name,
            layout=container.streams.audio[0].layout.name,
            rate=target_sampling_rate, #container.streams.audio[0].rate
        )

    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        if frame.pts == 0:
            pass
        else:
            if resampler != None:
                frame = resampler.resample(frame)[0]

        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    # result = result[1:] # remove the first audio frame due to less length
    # or padding the first audio frame due to less length??
    return result, max_pts

try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True

# from pytorchvideo
class ShortSideScale(nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return self.short_side_scale(
            x, self._size, self._interpolation, self._backend
        )

    def short_side_scale(
            self,
            x: torch.Tensor,
            size: int,
            interpolation: str = "bilinear",
            backend: str = "pytorch",
    ) -> torch.Tensor:
        """
        Determines the shorter spatial dim of the video (i.e. width or height) and scales
        it to the given size. To maintain aspect ratio, the longer side is then scaled
        accordingly.
        Args:
            x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
            size (int): The size the shorter side is scaled to.
            interpolation (str): Algorithm used for upsampling,
                options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
            backend (str): backend used to perform interpolation. Options includes
                `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
                differently on linear interpolation on some versions.
                https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
        Returns:
            An x-like Tensor with scaled spatial dims.
        """  # noqa
        assert len(x.shape) == 4
        assert x.dtype == torch.float32
        assert backend in ("pytorch", "opencv")
        c, t, h, w = x.shape
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        if backend == "pytorch":
            return torch.nn.functional.interpolate(
                x, size=(new_h, new_w), mode=interpolation, align_corners=False
            )
        elif backend == "opencv":
            return self._interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
        else:
            raise NotImplementedError(f"{backend} backend not supported.")

    @torch.jit.ignore
    def _interpolate_opencv(
            self, x: torch.Tensor, size: Tuple[int, int], interpolation: str
    ) -> torch.Tensor:
        """
        Down/up samples the input torch tensor x to the given size with given interpolation
        mode.
        Args:
            input (Tensor): the input tensor to be down/up sampled.
            size (Tuple[int, int]): expected output spatial size.
            interpolation: model to perform interpolation, options include `nearest`,
                `linear`, `bilinear`, `bicubic`.
        """
        if not _HAS_CV2:
            raise ImportError(
                "opencv is required to use opencv transforms. Please "
                "install with 'pip install opencv-python'."
            )

        _opencv_pytorch_interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        assert interpolation in _opencv_pytorch_interpolation_map
        new_h, new_w = size
        img_array_list = [
            img_tensor.squeeze(0).numpy()
            for img_tensor in x.permute(1, 2, 3, 0).split(1, dim=0)
        ]
        resized_img_array_list = [
            cv2.resize(
                img_array,
                (new_w, new_h),  # The input order for OpenCV is w, h.
                interpolation=_opencv_pytorch_interpolation_map[interpolation],
            )
            for img_array in img_array_list
        ]
        img_array = np.concatenate(
            [np.expand_dims(img_array, axis=0) for img_array in resized_img_array_list],
            axis=0,
        )
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_array))
        img_tensor = img_tensor.permute(3, 0, 1, 2)
        return img_tensor

class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return self.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"

    def normalize(self, clip, mean, std, inplace=False):
        """
        Args:
            clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
            mean (tuple): pixel RGB mean. Size is (3)
            std (tuple): pixel standard deviation. Size is (3)
        Returns:
            normalized clip (torch.tensor): Size is (C, T, H, W)
        """
        if not self._is_tensor_video_clip(clip):
            raise ValueError("clip should be a 4D torch.tensor")
        if not inplace:
            clip = clip.clone()
        mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return clip

    def _is_tensor_video_clip(self, clip):
        if not torch.is_tensor(clip):
            raise TypeError("clip should be Tensor. Got %s" % type(clip))

        if not clip.ndimension() == 4:
            raise ValueError("clip should be 4D. Got %dD" % clip.dim())

        return True

class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(self.uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(self.uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res

    def uniform_crop(self, images, size, spatial_idx, boxes=None, scale_size=None):
        """
        Perform uniform spatial sampling on the images and corresponding boxes.
        Args:
            images (tensor): images to perform uniform crop. The dimension is
                `num frames` x `channel` x `height` x `width`.
            size (int): size of height and weight to crop the images.
            spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
                is larger than height. Or 0, 1, or 2 for top, center, and bottom
                crop if height is larger than width.
            boxes (ndarray or None): optional. Corresponding boxes to images.
                Dimension is `num boxes` x 4.
            scale_size (int): optinal. If not None, resize the images to scale_size before
                performing any crop.
        Returns:
            cropped (tensor): images with dimension of
                `num frames` x `channel` x `size` x `size`.
            cropped_boxes (ndarray or None): the cropped boxes with dimension of
                `num boxes` x 4.
        """
        assert spatial_idx in [0, 1, 2]
        ndim = len(images.shape)
        if ndim == 3:
            images = images.unsqueeze(0)
        height = images.shape[2]
        width = images.shape[3]

        if scale_size is not None:
            if width <= height:
                width, height = scale_size, int(height / width * scale_size)
            else:
                width, height = int(width / height * scale_size), scale_size
            images = torch.nn.functional.interpolate(
                images,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        y_offset = int(math.ceil((height - size) / 2))
        x_offset = int(math.ceil((width - size) / 2))

        if height > width:
            if spatial_idx == 0:
                y_offset = 0
            elif spatial_idx == 2:
                y_offset = height - size
        else:
            if spatial_idx == 0:
                x_offset = 0
            elif spatial_idx == 2:
                x_offset = width - size
        cropped = images[:, :, y_offset: y_offset + size, x_offset: x_offset + size]
        cropped_boxes = self.crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
        if ndim == 3:
            cropped = cropped.squeeze(0)
        return cropped, cropped_boxes

    def crop_boxes(self, boxes, x_offset, y_offset):
        """
        Perform crop on the bounding boxes given the offsets.
        Args:
            boxes (ndarray or None): bounding boxes to perform crop. The dimension
                is `num boxes` x 4.
            x_offset (int): cropping offset in the x axis.
            y_offset (int): cropping offset in the y axis.
        Returns:
            cropped_boxes (ndarray or None): the cropped boxes with dimension of
                `num boxes` x 4.
        """
        cropped_boxes = boxes.copy()
        cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
        cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

        return cropped_boxes

class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return self.uniform_temporal_subsample(
            x, self._num_samples, self._temporal_dim
        )


    def uniform_temporal_subsample(
            self, x: torch.Tensor, num_samples: int, temporal_dim: int = -3
    ) -> torch.Tensor:
        """
        Uniformly subsamples num_samples indices from the temporal dimension of the video.
        When num_samples is larger than the size of temporal dimension of the video, it
        will sample frames based on nearest neighbor interpolation.

        Args:
            x (torch.Tensor): A video tensor with dimension larger than one with torch
                tensor type includes int, long, float, complex, etc.
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.

        Returns:
            An x-like Tensor with subsampled temporal dimension.
        """
        t = x.shape[temporal_dim]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, temporal_dim, indices)

if __name__ == "__main__":
    import cProfile
    env_db = lmdb.open('/home/lrh/NewDisk/MSA_Datasets/SemanticMSA_Datasets/MOSI_lmdb', readonly=True,
                                create=False)  # readahead=not_check_distributed())
    txn = env_db.begin(buffers=True)
    load_audiovision_lmdb(txn,'8d-gEyoeBzc$_$1')
    # cProfile.run("load_audiovision_lmdb(txn,'8d-gEyoeBzc$_$20')")
