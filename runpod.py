import argparse
import os
import io
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus

import soundfile as sf
import uvicorn
import pyrootutils

# Setup root path so imports work correctly
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fastapi import FastAPI, HTTPException, Request, Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from tools.server.model_manager import ModelManager
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference
from tools.server.api_utils import get_content_type, inference_async

# Global ModelManager instance
model_manager = None
args = None

def parse_args():
    parser = argparse.ArgumentParser(description="RunPod FastAPI Server for Fish Speech")
    parser.add_argument("--mode", type=str, default="tts", choices=["tts"])
    parser.add_argument("--llama-checkpoint-path", type=str, default="checkpoints/s2-pro")
    parser.add_argument("--decoder-checkpoint-path", type=str, default="checkpoints/s2-pro/codec.pth")
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument("--compile", action="store_true", help="Compile the model for faster inference")
    parser.add_argument("--listen", type=str, default="0.0.0.0:8000", help="Listen address (host:port)")
    
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, args
    args = parse_args()
    
    # Environment variables override CLI arguments for easier container configuration
    mode = os.getenv("MODE", args.mode)
    device = os.getenv("DEVICE", args.device)
    half = os.getenv("HALF", str(args.half)).lower() in ("true", "1", "yes")
    compile_flag = os.getenv("COMPILE", str(args.compile)).lower() in ("true", "1", "yes")
    llama_checkpoint_path = os.getenv("LLAMA_CHECKPOINT_PATH", args.llama_checkpoint_path)
    decoder_checkpoint_path = os.getenv("DECODER_CHECKPOINT_PATH", args.decoder_checkpoint_path)
    decoder_config_name = os.getenv("DECODER_CONFIG_NAME", args.decoder_config_name)
    
    logger.info("Initializing ModelManager for serverless environment...")
    logger.info(f"Config: mode={mode}, device={device}, half={half}, compile={compile_flag}")
    logger.info(f"Paths: llama={llama_checkpoint_path}, decoder={decoder_checkpoint_path}")
    
    try:
        model_manager = ModelManager(
            mode=mode,
            device=device,
            half=half,
            compile=compile_flag,
            llama_checkpoint_path=llama_checkpoint_path,
            decoder_checkpoint_path=decoder_checkpoint_path,
            decoder_config_name=decoder_config_name
        )
        logger.info("ModelManager initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}")
        traceback.print_exc()
        raise e
        
    yield
    
    logger.info("Shutting down model server...")
    model_manager = None

app = FastAPI(
    title="Fish Speech RunPod Serverless API",
    description="Lightweight FastAPI server for serverless deployment of Fish Speech",
    version="1.5.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "statusCode": exc.status_code,
                "message": exc.detail,
                "error": HTTPStatus(exc.status_code).phrase
            }
        )
        
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content={
                "statusCode": HTTPStatus.UNPROCESSABLE_ENTITY,
                "message": exc.errors(),
                "error": "Validation Error"
            }
        )

    logger.error(f"Unhandled exception: {exc}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={
            "statusCode": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": str(exc),
            "error": "Internal Server Error"
        }
    )

@app.get("/ping")
@app.post("/ping")
async def ping():
    if model_manager is None or model_manager.tts_inference_engine is None:
        return JSONResponse(
            content={"status": "initializing"},
            status_code=HTTPStatus.NO_CONTENT
        )
    return {
        "status": "healthy",
        "device": model_manager.device,
        "mode": model_manager.mode,
    }

@app.post("/tts")
async def inference_endpoint(request: Request):
    if model_manager is None or model_manager.tts_inference_engine is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet"
        )
        
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Only application/json content type is supported"
        )
        
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Failed to parse JSON body: {str(e)}"
        )
        
    try:
        req = ServeTTSRequest(**data)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid request schema: {str(e)}"
        )

    # Discard caching behaviours as this is a serverless environment
    req.use_memory_cache = "off"
    engine = model_manager.tts_inference_engine
    
    # Get sample rate
    if hasattr(engine.decoder_model, "spec_transform"):
        sample_rate = engine.decoder_model.spec_transform.sample_rate
    else:
        sample_rate = engine.decoder_model.sample_rate

    try:
        if req.streaming:
            if req.format != "wav":
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="Streaming only supports WAV format"
                )
            
            return StreamingResponse(
                inference_async(req, engine),
                media_type=get_content_type(req.format),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                }
            )
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
            return Response(
                content=audio_bytes,
                media_type=get_content_type(req.format),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in speech generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate speech: {str(e)}"
        )


if __name__ == "__main__":
    args = parse_args()
    
    # Parse listen host and port
    if ":" in args.listen:
        host, port = args.listen.split(":")
        port = int(port)
    else:
        host = args.listen
        port = 8000
        
    # Support PORT env var for serverless runtime
    env_port = os.getenv("PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            logger.warning(f"Invalid PORT env var: {env_port}, defaulting to {port}")
        
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
