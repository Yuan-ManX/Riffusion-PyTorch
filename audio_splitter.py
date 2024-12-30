import shutil
import subprocess
import tempfile
import typing as T
from pathlib import Path
import numpy as np
import pydub

import torch
import torchaudio
from torchaudio.transforms import Fade

from util import audio_util


def split_audio(
    segment: pydub.AudioSegment,
    model_name: str = "htdemucs_6s",
    extension: str = "wav",
    jobs: int = 4,
    device: str = "cuda",
) -> T.Dict[str, pydub.AudioSegment]:
    """
    使用 demucs 将音频分割成多个音轨（stems）。
    
    参数:
        segment (pydub.AudioSegment): 要分割的音频片段。
        model_name (str): 使用的 demucs 模型名称。默认值为 "htdemucs_6s"。
        extension (str): 输出音频文件的扩展名。默认值为 "wav"。
        jobs (int): 并行作业数。默认值为 4。
        device (str): 使用的设备，"cuda" 或 "cpu"。默认值为 "cuda"。
    
    返回:
        Dict[str, pydub.AudioSegment]: 分割后的音轨字典，键为音轨名称，值为对应的音频片段。
    """
    # 创建一个临时目录用于存储中间文件
    tmp_dir = Path(tempfile.mkdtemp(prefix="split_audio_"))

    # 将输入的音频片段保存为临时 MP3 文件
    audio_path = tmp_dir / "audio.mp3"
    segment.export(audio_path, format="mp3")

    # 组装运行 demucs 的命令列表
    command = [
        "demucs",
        str(audio_path),
        "--name",
        model_name,
        "--out",
        str(tmp_dir),
        "--jobs",
        str(jobs),
        "--device",
        device if device != "mps" else "cpu",
    ]
    print(" ".join(command))

    # 如果输出文件扩展名为 "mp3"，则添加 "--mp3" 参数
    if extension == "mp3":
        command.append("--mp3")

    # 运行 demucs 命令
    subprocess.run(
        command,
        check=True,
    )

    # 加载分割后的音轨
    stems = {}
    # 遍历输出目录中所有符合扩展名的音轨文件
    for stem_path in tmp_dir.glob(f"{model_name}/audio/*.{extension}"):
        stem = pydub.AudioSegment.from_file(stem_path)
        stems[stem_path.stem] = stem

    # 删除临时目录及其中的所有文件
    shutil.rmtree(tmp_dir)

    return stems


class AudioSplitter:
    """
    将音频分割成乐器音轨，如鼓、低音、歌声等。

    注意(hayk): 这个类已经被弃用，因为其性能不如 demucs 仓库中新的混合Transformer模型。
    请参见上面的函数。可能在未来直接删除这个类。

    参见:
        - demucs 仓库中的混合Transformer模型
    """


    def __init__(
        self,
        segment_length_s: float = 10.0,
        overlap_s: float = 0.1,
        device: str = "cuda",
    ):
        """
        初始化 AudioSplitter 类。

        参数:
            segment_length_s (float): 每个分割片段的长度（秒）。默认值为 10.0 秒。
            overlap_s (float): 分割片段之间的重叠时间（秒）。默认值为 0.1 秒。
            device (str): 使用的设备，"cuda" 或 "cpu"。默认值为 "cuda"。
        """
        self.segment_length_s = segment_length_s
        self.overlap_s = overlap_s
        self.device = device

        self.model = self.load_model().to(device)

    @staticmethod
    def load_model(model_path: str = "models/hdemucs_high_trained.pt") -> torchaudio.models.HDemucs:
        """
        加载预训练的 HDEMUCS PyTorch 模型。

        参数:
            model_path (str): 模型文件的路径。默认值为 "models/hdemucs_high_trained.pt"。

        返回:
            torchaudio.models.HDemucs: 加载并配置好的 HDEMUCS 模型。
        """
        # 注意(hayk): 音源已经嵌入在预训练模型中，无法更改
        # 初始化 HDEMUCS 模型，指定音源为 ["drums", "bass", "other", "vocals"]
        model = torchaudio.models.hdemucs_high(sources=["drums", "bass", "other", "vocals"])

        # 下载模型文件并获取本地路径
        path = torchaudio.utils.download_asset(model_path)
        # 加载模型的 state_dict
        state_dict = torch.load(path)
        # 将 state_dict 加载到模型中
        model.load_state_dict(state_dict)
        # 设置模型为评估模式
        model.eval()

        return model

    def split(self, audio: pydub.AudioSegment) -> T.Dict[str, pydub.AudioSegment]:
        """
        将给定的音频片段分割成乐器音轨。

        参数:
            audio (pydub.AudioSegment): 要分割的音频片段。

        返回:
            Dict[str, pydub.AudioSegment]: 分割后的乐器音轨字典，键为音轨名称，值为对应的音频片段。
        """
        # 如果音频是单声道，则转换为立体声
        if audio.channels == 1:
            audio_stereo = audio.set_channels(2)
        elif audio.channels == 2:
            audio_stereo = audio
        else:
            raise ValueError(f"Audio must be stereo, but got {audio.channels} channels")

        # 将音频转换为 (样本数, 声道数) 的浮点型 NumPy 数组
        waveform_np = np.array(audio_stereo.get_array_of_samples())
        waveform_np = waveform_np.reshape(-1, audio_stereo.channels)
        waveform_np_float = waveform_np.astype(np.float32)

        # 转换为 PyTorch 张量并调整通道顺序为 (声道数, 样本数)
        waveform = torch.from_numpy(waveform_np_float).to(self.device)
        waveform = waveform.transpose(1, 0)

        # 归一化
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()

        # 分割音频
        sources = self.separate_sources(
            waveform[None],
            sample_rate=audio.frame_rate,
        )[0]

        # 反归一化
        sources = sources * ref.std() + ref.mean()

        # 转换为 NumPy 数组
        sources_np = sources.cpu().numpy().astype(waveform_np.dtype)

        # 将 NumPy 数组转换为 pydub AudioSegment 对象
        stem_segments = [
            audio_util.audio_from_waveform(waveform, audio.frame_rate) for waveform in sources_np
        ]

        # 如果原始音频是单声道，则将分割后的音轨也转换为单声道
        if audio.channels == 1:
            stem_segments = [stem.set_channels(1) for stem in stem_segments]

        # 将音轨名称和对应的音频片段组成字典
        return dict(zip(self.model.sources, stem_segments))

    def separate_sources(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 44100,
    ):
        """
        将模型应用到给定的波形中，以分割音源。使用渐变和重叠来平滑边缘。

        参数:
            waveform (torch.Tensor): 输入的波形张量，形状为 (批次大小, 声道数, 样本数)。
            sample_rate (int): 采样率。默认值为 44100。

        返回:
            torch.Tensor: 分割后的音源张量，形状为 (批次大小, 音源数, 声道数, 样本数)。
        """
        batch, channels, length = waveform.shape

        # 计算每个分割片段的长度（样本数），包括重叠部分
        chunk_len = int(sample_rate * self.segment_length_s * (1 + self.overlap_s))

        # 初始化起始和结束位置
        start = 0
        end = chunk_len

        # 计算重叠的样本数
        overlap_frames = self.overlap_s * sample_rate
        # 初始化 Fade 对象，用于处理渐变
        fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

        # 初始化最终输出张量，形状为 (批次大小, 音源数, 声道数, 样本数)
        final = torch.zeros(batch, len(self.model.sources), channels, length, device=self.device)

        # 循环处理每个分割片段
        while start < length - overlap_frames:
            # 获取当前片段
            chunk = waveform[:, :, start:end]
            # 使用模型进行音源分割
            with torch.no_grad():
                out = self.model.forward(chunk)
            # 应用渐变
            out = fade(out)
            # 将输出添加到最终结果中
            final[:, :, :, start:end] += out
            # 处理渐变长度
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            # 更新结束位置
            end += chunk_len
            # 如果结束位置超过总长度，则取消渐变
            if end >= length:
                fade.fade_out_len = 0

        return final
