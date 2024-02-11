from transformers import AutoProcessor, SeamlessM4Tv2Model
import sounddevice as sd
import numpy as np
from datasets import load_dataset


def play_audio(audio_array: np.ndarray, sampling_rate: int):
    sd.play(audio_array, samplerate=sampling_rate)
    sd.wait()


def speech_to_text(audio: np.ndarray) -> str:
    audio_inputs = processor(audios=audio, return_tensors="pt", padding="longest")
    output_tokens = model.generate(**audio_inputs, tgt_lang="eng")
    transcription = processor.decode(output_tokens, skip_special_tokens=True)
    return transcription


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
audio_sample = next(iter(dataset))["audio"]

audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt", sampling_rate=16000)
audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
# play_audio(audio_array_from_audio, 16000)
print(speech_to_text(audio_sample["array"]))