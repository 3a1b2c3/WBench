# Choosing the VLM judge

The VLM metrics (`--phase vlm`) call a vision-language model through `src/metrics/vlm/vlm_evaluator.py`. The default is Doubao-Seed-2.0-lite on Volcengine ARK, which is what the published leaderboard scores use. An NVIDIA-hosted judge can be used instead. Scores from any other judge are **not comparable to the leaderboard**; use them only to compare your own runs with each other.

## Which judge is used

| Environment | Judge |
| --- | --- |
| nothing set | ARK endpoint and `doubao-seed-2-0-lite-260215` (needs `VLM_API_KEY`) |
| `VLM_API_KEY` set | ARK and Doubao, unless `VLM_API_URL` / `VLM_MODEL_NAME` override them |
| only `NVIDIA_API_KEY` set | NVIDIA endpoint and model below, with a warning that scores are not leaderboard-comparable |
| any other model name | rejected unless `VLM_ALLOW_CUSTOM_MODEL=1` |

If both keys are set, `VLM_API_KEY` wins. Keys come from the environment only; they are never printed or stored.

NVIDIA defaults (class constants in `VLMClient`):

- URL: `https://inference-api.nvidia.com/v1/chat/completions`. The code notes that `integrate.api.nvidia.com` was returning gateway errors for every model when this was last touched.
- Model: `nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.

`VLM_API_URL` and `VLM_MODEL_NAME` override both. The URL must contain `chat/completions` to use the OpenAI request format; any other URL is treated as ARK's Responses API.

## Request differences for non-Doubao models

- The Doubao-only `"thinking": {"type": "disabled"}` field is sent only to the two allowed Doubao models.
- For other models the request sets `chat_template_kwargs: {"enable_thinking": false}`. The yes/no questions allow only 10 output tokens, and a reasoning model that thinks first spends them all and returns no answer. The code comments say the default model was confirmed live with this setting, accepting a `video_url` and answering "yes"/"no" with no reasoning tokens.
- The clip is sent as one base64 `video_url`. If an endpoint accepts only images, the client falls back to frames when no video is given, which changes what the judge sees.

## Checking a judge before a long run

```
python tools/test_vlm_judge.py path/to/clip.mp4
```

It prints the URL and model, then asks one yes/no question that should be answered yes and a control question that should be answered no, in video mode and in image-frame mode (`--frames 0` skips the second). Use a short clip with visible content. It exits 1 if either mode fails, including when the reply is empty. Run it in the same shell and with the same environment variables as the real run.

Examples (PowerShell):

```
$env:NVIDIA_API_KEY = "<key>"
```
```
python tools/test_vlm_judge.py C:\path\to\clip.mp4
```

Clear any `VLM_API_URL` / `VLM_MODEL_NAME` left over from earlier attempts first, or the test will talk to the wrong service.
