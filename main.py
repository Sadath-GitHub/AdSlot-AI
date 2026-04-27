import os
import json
import tempfile
import uvicorn
import anthropic
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI(
    title="Ad Timestamp Finder API",
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
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded.")

claude_client = anthropic.Anthropic()


def transcribe_video(video_path: str) -> list[dict]:
    segments, _ = whisper_model.transcribe(video_path)
    return [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        }
        for seg in segments
    ]


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def find_ad_timestamps(segments: list[dict], video_duration: float, ad_count: int = 3) -> dict:
    transcript_text = "\n".join(
        f"[{seg['start']}s - {seg['end']}s]: {seg['text']}"
        for seg in segments
    )

    # Dynamic buffer: at most 10% of video duration, capped at 30s
    buffer = min(30, video_duration * 0.10)
    valid_start = round(buffer, 1)
    valid_end   = round(video_duration - buffer, 1)

    prompt = f"""You are an expert video ad placement strategist.

Below is a timestamped transcript of a video (total duration: {round(video_duration, 1)} seconds).

TRANSCRIPT:
{transcript_text}

YOUR TASK: Return EXACTLY {ad_count} ad timestamp(s). No more, no less. This is mandatory.

Rules:
- Only place ads between {valid_start}s and {valid_end}s
- Prefer natural pauses, topic transitions, or low-information-density moments
- Spread the timestamps as evenly as possible across the valid window
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
}}

Remember: the array MUST contain exactly {ad_count} item(s)."""

    message = claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    result = json.loads(response_text)

    # Safety net: if Claude still returns wrong count, fill or trim
    timestamps = result.get("ad_timestamps", [])

    # Trim if too many
    timestamps = timestamps[:ad_count]

    # Fill if too few by evenly spacing across valid window
    if len(timestamps) < ad_count:
        existing_secs = {t["timestamp_seconds"] for t in timestamps}
        interval = (valid_end - valid_start) / (ad_count + 1)
        for i in range(1, ad_count + 1):
            if len(timestamps) >= ad_count:
                break
            candidate = round(valid_start + i * interval, 1)
            # Avoid duplicating existing timestamps (within 5s)
            if all(abs(candidate - e) > 5 for e in existing_secs):
                timestamps.append({
                    "timestamp_seconds": candidate,
                    "timestamp_formatted": format_timestamp(candidate),
                    "reason": "Evenly distributed placement across video duration."
                })
                existing_secs.add(candidate)

    result["ad_timestamps"] = timestamps
    return result


@app.post("/find-ad-timestamps")
async def find_ad_timestamps_endpoint(
    video: UploadFile = File(...),
    ad_count: int = Form(3)
):
    ad_count = max(1, min(ad_count, 10))

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
