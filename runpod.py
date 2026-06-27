import io
import os
import base64
import runpod
import soundfile as sf

from loguru import logger
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager


model_manager = None

device = os.getenv("DEVICE", "cuda")
half = os.getenv("HALF", "1") == "1"
compile_flag = os.getenv("COMPILE", "1") == "1"
llama_checkpoint_path = os.getenv("LLAMA_CHECKPOINT_PATH", "checkpoints/s2-pro")
decoder_checkpoint_path = os.getenv(
    "DECODER_CHECKPOINT_PATH", "checkpoints/s2-pro/codec.pth"
)
decoder_config_name = os.getenv("DECODER_CONFIG_NAME", "modded_dac_vq")


def handler(job):
    global model_manager

    if model_manager is None:
        logger.info("Initializing ModelManager for serverless environment...")

        model_manager = ModelManager(
            mode="tts",
            device=device,
            half=half,
            compile=compile_flag,
            llama_checkpoint_path=llama_checkpoint_path,
            decoder_checkpoint_path=decoder_checkpoint_path,
            decoder_config_name=decoder_config_name,
        )

        logger.info("ModelManager initialized successfully!")

    job_input = job["input"]
    req = ServeTTSRequest(**job_input)

    engine = model_manager.tts_inference_engine

    if hasattr(engine.decoder_model, "spec_transform"):
        sample_rate = engine.decoder_model.spec_transform.sample_rate
    else:
        sample_rate = engine.decoder_model.sample_rate

    try:
        if req.streaming:
            if req.format != "wav":
                yield {"error": "Streaming only supports WAV format"}
                return

            # Streaming generation
            for chunk in inference(req, engine):
                if isinstance(chunk, bytes):
                    yield {
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                        "format": req.format,
                        "sample_rate": sample_rate,
                    }
        else:
            # Sync generation
            gen = inference(req, engine)
            fake_audios = next(gen)

            buffer = io.BytesIO()
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

            audio_bytes = buffer.getvalue()
            yield {
                "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": req.format,
                "sample_rate": sample_rate,
            }
    except Exception as e:
        logger.error(f"Error in speech generation: {e}", exc_info=True)
        yield {"error": f"Failed to generate speech: {str(e)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
