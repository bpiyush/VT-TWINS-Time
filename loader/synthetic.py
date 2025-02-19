"""Synthetic Dataset object."""
from glob import glob
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import pickle
import torch
import warnings
warnings.simplefilter("ignore", UserWarning)


class Synthetic(Dataset):
    """MSRVTT Video-Text loader."""

    def __init__(
            self,
            metadata_dir="",
            video_root="",
            text_key="caption",
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='../data/dict.npy',
            max_words=30,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        
        metadata_files = sorted(glob(os.path.join(metadata_dir, "*.pt")))
        metadata = [torch.load(f) for f in metadata_files]
        sentences = [m[text_key] for m in metadata]
        video_ids = sorted([os.path.basename(f).split(".pt")[0] for f in metadata_files])
        self.data = pd.DataFrame({"video_id": video_ids, "sentence": sentences})
        
        # print an example
        print()
        print("Video ID: ", self.data['video_id'].values[0])
        print("Sentence: ", self.data['sentence'].values[0])
        print()

        self.video_root = video_root

        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        self.word_to_token = {}
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def __len__(self):
        return len(self.data)

    def _get_video(self, video_path, start, end, num_clip):
        video = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(start, max(start, end-self.num_sec - 0.4), num_clip) 
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s) 
        return video

    def _get_video_start(self, video_path, start):
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _get_duration(self, video_path):
        probe = ffmpeg.probe(video_path)
        return probe['format']['duration']

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        cap = self.data['sentence'].values[idx]
        video_path = os.path.join(self.video_root, video_id + '.mp4')
        duration = self._get_duration(video_path)
        text = self.words_to_ids(cap)
        video = self._get_video(video_path, 0, float(duration), self.num_clip)
        return {'video': video, 'text': text}


if __name__ == "__main__":
    metadata_dir = "/ssd/pbagad/datasets/ToT-syn-v2.0/metadata"
    video_root = "/ssd/pbagad/datasets/ToT-syn-v2.0/videos"
    dataset = Synthetic(metadata_dir, video_root)
    
    instance = dataset[0]
    assert instance["video"].shape == torch.Size([4, 3, 32, 224, 224])
    assert instance["text"].shape == torch.Size([30])
    