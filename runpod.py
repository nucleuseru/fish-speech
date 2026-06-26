import argparse
import base64
import io
import os
import traceback

import pyrootutils
import soundfile as sf

from loguru import logger
import runpod

from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="RunPod Serverless Handler for Fish Speech"
    )
    parser.add_argument("--mode", type=str, default="tts", choices=["tts"])
    parser.add_argument(
        "--llama-checkpoint-path", type=str, default="checkpoints/s2-pro"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model for faster inference",
    )

    parsed_args, _ = parser.parse_known_args()
    return parsed_args


# Global model initialization at module load time
args = parse_args()

# Environment variables override CLI arguments for easier container configuration
mode = os.getenv("MODE", args.mode)
device = os.getenv("DEVICE", args.device)
half = os.getenv("HALF", str(args.half)).lower() in ("true", "1", "yes")
compile_flag = os.getenv("COMPILE", str(args.compile)).lower() in (
    "true",
    "1",
    "yes",
)
llama_checkpoint_path = os.getenv("LLAMA_CHECKPOINT_PATH", args.llama_checkpoint_path)
decoder_checkpoint_path = os.getenv(
    "DECODER_CHECKPOINT_PATH", args.decoder_checkpoint_path
)
decoder_config_name = os.getenv("DECODER_CONFIG_NAME", args.decoder_config_name)

logger.info("Initializing ModelManager for serverless environment...")
logger.info(
    f"Config: mode={mode}, device={device}, half={half}, compile={compile_flag}"
)
logger.info(f"Paths: llama={llama_checkpoint_path}, decoder={decoder_checkpoint_path}")

try:
    model_manager = ModelManager(
        mode=mode,
        device=device,
        half=half,
        compile=compile_flag,
        llama_checkpoint_path=llama_checkpoint_path,
        decoder_checkpoint_path=decoder_checkpoint_path,
        decoder_config_name=decoder_config_name,
    )
    logger.info("ModelManager initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize ModelManager: {e}")
    traceback.print_exc()
    raise e


def handler(job):
    job_input = job["input"]
    try:
        req = ServeTTSRequest(**job_input)
    except Exception as e:
        yield {"error": f"Invalid request schema: {str(e)}"}
        return

    engine = model_manager.tts_inference_engine
    if engine is None:
        yield {"error": "Model is not loaded yet"}
        return

    # Get sample rate
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
