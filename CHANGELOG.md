# Changelog: gemini-linux-fix branch

All changes relative to the `main` branch.

## Linux / Cross-Platform Compatibility

- **OS path separators**: Replaced all hardcoded Windows `\\` path separators with `os.path.join()` across `config_loader.py`, `setup.py`, `stt.py`, `image_manager.py`, and `game_definitions.py`.
- **Removed Windows-only dependencies**: Removed `pywin32` from `requirements.txt`; gated `win32gui` / `ctypes` imports behind `platform.system() == "Windows"` checks in `image_manager.py`.
- **Moonshine optional import**: Wrapped `moonshine_onnx` import in try/except so Mantella can run without it installed (raises a clear `ImportError` only if Moonshine STT is selected).
- **Unpinned pandas version**: Changed `pandas==1.5.3` to `pandas` to allow compatible versions on all platforms.
- **Piper binary on POSIX**: In `piper.py`, uses `piper` instead of `piper.exe` on POSIX systems and auto-sets execute permission if missing.
- **xVASynth case-insensitive model paths**: Falls back to lowercase model folder name on case-sensitive filesystems (Linux).
- **Screenshot capture on Linux**: Uses `mss` primary monitor capture instead of `win32gui` window detection.
- **Temp directory handling**: Uses `TMPDIR` or `/tmp` fallback when `TMP` environment variable is not set (Linux).
- **Lip sync tools gated**: Lip generation tools (LipGen, FaceFXWrapper, LipFuzer) are skipped on non-Windows platforms with a clear log message.

## Audio Device & Sample Rate Support

- **Audio input device selector**: Added UI dropdown (`Speech-to-Text` -> `Audio Input Device`) that enumerates available input devices via `sounddevice`. Supports selection by index, name substring, or dropdown format.
- **Native sample rate resampling**: Records at the audio device's native sample rate (e.g. 48kHz for USB audio interfaces) and resamples to 16kHz using `scipy.signal.resample_poly` before feeding to VAD and STT. Fixes `paInvalidSampleRate` errors with pro audio interfaces.
- **Device fallback**: If the selected audio device fails to open (common on Linux with PipeWire/ALSA conflicts), automatically falls back to the system default device.
- **No-default-device handling**: When PortAudio reports no default input device (index -1), enumerates all devices to find the first available input.
- **Input overflow resilience**: Audio chunks with PortAudio "input overflow" status are no longer discarded; they contain valid data and discarding them created gaps that confused the VAD.
- **Reduced overflow frequency**: Removed `latency='low'` from audio stream to use default latency, reducing overflow occurrences with USB audio interfaces.

## Speech-to-Text Improvements

- **Whisper hallucination filtering**: Added `_is_hallucination()` method that detects and silently discards:
  - Prompt echoes (Whisper repeating "This is a conversation with..." from its `initial_prompt`)
  - Repetitive phrases (same sentence repeated multiple times)
  - Added `'you'` to the static ignore list
- **NPC audio echo filtering**: Registers recent NPC voicelines and compares incoming transcriptions against them using word-level similarity. Filters out transcriptions caused by the microphone picking up game audio from speakers (e.g. NPC says "Stay in the house!" -> mic picks it up -> Whisper transcribes "Stay in the house." -> filtered).
- **Diagnostics native rate support**: `test_microphone()` and `test_stt_live()` in diagnostics now use the device's native sample rate with resampling, matching the main STT pipeline.

## Reliability & Error Handling

- **Graceful missing voice model handling**: `VoiceModelNotFound` exceptions during voice model preloading no longer crash the entire ASGI stack. Instead, a warning is logged and the conversation continues. The voice model will attempt to load again during synthesis (which is already wrapped in error handling).
- **LLM function call sanitization**: Raw LLM function-call markup (`<function=...>`, `<parameter=...>`, `<tool_call>`) is stripped from voicelines before TTS synthesis, preventing spoken function calls.
- **Action response timeout**: When the game mod doesn't respond to a `requires_response=True` action (e.g. `reportcrime`, `absolve_crime`) within 30 seconds, the conversation auto-resumes instead of hanging indefinitely. Player input during the wait also clears the action flag to prevent competing LLM generations.
- **Clean process shutdown**: Added SIGINT/SIGTERM signal handlers with a 3-second grace period and force-exit fallback. Fixes the process becoming unkillable (requiring `kill -9`) after running with microphone input for extended periods. Also added a timeout to the STT processing thread join to prevent shutdown hangs.
- **LLM tool use fallback**: If the selected model doesn't support tool use (OpenRouter 404 "No endpoints found that support tool use"), the API call is automatically retried without tools. The conversation continues normally without in-game actions for that exchange.
- **max_tokens auto-recovery**: If the model rejects `max_tokens` as below its minimum (e.g. reasoning models like GPT-5.2), the call is retried without the `max_tokens` limit. Diagnostics test completion also handles this gracefully.

## Console Logging

- **Diagnostics test logging**: Each diagnostic test (LLM connection, TTS service, STT service, microphone, STT live) now logs its start and OK/FAILED result to the console.
- **Config logging**: Logs when `config.ini` is loaded, a summary of key settings on initial load (game, LLM, TTS, STT, mod path), and when config values change.
- **Game/conversation lifecycle**: Logs conversation start (with world ID, mic/ptt status), conversation end (with NPC names), character loading (with name, base ID, race), and player interruptions.
- **STT/TTS events**: Logs Transcriber initialization (service, language), audio device selection and native rate, and voice model loading in Piper.
- **Overflow log level reduced**: Audio stream overflow warnings downgraded from WARNING to DEBUG to reduce console noise.

## LLM Settings

- **Reasoning effort toggle**: Added `LLM` -> `Reasoning Effort` dropdown in the UI (Default / Low / Medium / High). Controls how much effort reasoning-capable models (OpenAI o-series, DeepSeek R1, etc.) spend on internal thinking before responding. Ignored by models that don't support it.

## Diagnostics Module

- **New `src/ui/diagnostics.py`**: Added comprehensive diagnostics module with tests for LLM connection, TTS service, STT service, microphone input, and live STT transcription. All tests produce structured results and console logs.
