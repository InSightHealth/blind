import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset



class Speech_To_Text:
    def __init__(self) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./whisper-large-v3", low_cpu_mem_usage=True, use_safetensors=True, device_map='cuda')
        
        self.processor = AutoProcessor.from_pretrained('./whisper-large-v3')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.pipe = pipeline(
        "automatic-speech-recognition",
        model=self.model,
        tokenizer=self.processor.tokenizer,
        feature_extractor=self.processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=self.torch_dtype,
        device=self.device,
        )
        
    def speech_to_text(self, mp3):
        result = self.pipe(mp3, return_timestamps=False)
        return result['text']
