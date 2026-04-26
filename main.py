import os
import json
import tempfile
import anthropic
import whisper
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AdSlot AI",
    description="Upload a video and get the best timestamps to place ads using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

claude_client = anthropic.Anthropic(api_key="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")


def transcribe_video(video_path: str) -> list[dict]:
    result = whisper_model.transcribe(video_path)
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        }
        for seg in result["segments"]
    ]


def find_ad_timestamps(segments: list[dict], video_duration: float, ad_count) -> dict:
    transcript_text = "\n".join(
        f"[{seg['start']}s - {seg['end']}s]: {seg['text']}"
        for seg in segments
    )

    prompt = f"""You are an expert video ad placement strategist.
        Below is a timestamped transcript of a video (total duration: {round(video_duration, 1)} seconds).
    TRANSCRIPT:
{transcript_text}

Identify the {ad_count} best timestamp(s) to insert a short ad (15-30 seconds).

Criteria for a good ad timestamp:
- Natural pauses, topic transitions
- Low information density (filler, recap, or summary moments)
- Not in the middle of an important explanation or punchline
- Spread timestamps evenly across the video duration where possible
- If ideal spots are limited, still return EXACTLY {ad_count} timestamps at the best available positions
- Each timestamp must be unique (no duplicates)

Respond ONLY with valid JSON, no extra text, no markdown fences:
{{
  "ad_timestamps": [
    {{
      "timestamp_seconds": <number>,
      "timestamp_formatted": "<MM:SS>",
      "reason": "<one sentence explanation>"
    }}
  ],
  "summary": "<one sentence about the overall content of the video>"
}}"""

    message = claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    return json.loads(response_text)


@app.post("/find-ad-timestamps")
async def find_ad_timestamps_endpoint(
    video: UploadFile = File(...),
    ad_count: int = Form(3)
):
    ad_count = max(1, min(ad_count, 10))

    print(f"ad_count: {ad_count}")

    allowed_types = [
        "video/mp4", "video/quicktime", "video/x-msvideo",
        "video/webm", "video/mpeg"
    ]
    if video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{video.content_type}'. Use mp4, mov, avi, webm, or mpeg."
        )

    suffix = os.path.splitext(video.filename)[-1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        segments = transcribe_video(tmp_path)

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any speech from the video."
            )

        video_duration = segments[-1]["end"]
        result = find_ad_timestamps(segments, video_duration, ad_count)

        return JSONResponse(content={
            "filename": video.filename,
            "video_duration_seconds": video_duration,
            "transcript_segments": len(segments),
            "ad_count_requested": ad_count,
            **result
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}