from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TForSpeechToText
import sounddevice as sd
import numpy as np
from datasets import load_dataset


def play_audio(audio_array: np.ndarray, sampling_rate: int):
    sd.play(audio_array, samplerate=sampling_rate)
    sd.wait()


def speech_to_text(audio: np.ndarray) -> str:
    model = SeamlessM4TForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
    print(f"Model sampling rate: {model.config.sampling_rate}")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    play_audio(audio, model.config.sampling_rate)
    audio_inputs = processor(audios=audio, return_tensors="pt", padding="longest",
                             sampling_rate=model.config.sampling_rate)
    output_tokens = model.generate(**audio_inputs, tgt_lang="eng")
    transcription = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return transcription


if __name__ == "__main__":
    # dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
    # english audio dataset with 16k sampling rate
    dataset = load_dataset("google/fleurs", "en_us", split='test', streaming=True)
    audio_sample = next(iter(dataset))['audio']

    print(speech_to_text(audio_sample["array"]))
