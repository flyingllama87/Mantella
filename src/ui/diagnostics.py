import os
import sys
import platform
from pathlib import Path

import numpy as np
from src.config.config_loader import ConfigLoader
from src.config.definitions.tts_definitions import TTSEnum
import src.utils as utils

try:
    from moonshine_onnx import MoonshineOnnxModel
except ImportError:
    MoonshineOnnxModel = None

logger = utils.get_logger()

ON_POSIX = 'posix' in sys.builtin_module_names


class DiagnosticsRunner:
    """Runs connectivity and configuration diagnostics for the Mantella UI."""

    def __init__(self, config: ConfigLoader) -> None:
        self._config = config

    def _refresh_config(self):
        """Pick up any UI changes before running tests."""
        try:
            self._config.update_config_loader_with_changed_config_values()
        except Exception as e:
            logger.warning(f"Could not refresh config before diagnostics: {e}")

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_platform_info(self) -> str:
        lines = [
            "### Platform Info",
            f"- **OS:** {platform.system()} {platform.release()} ({platform.machine()})",
            f"- **Python:** {platform.python_version()}",
            f"- **POSIX:** {ON_POSIX}",
        ]
        return "\n".join(lines)

    def test_llm_connection(self) -> str:
        """Verify LLM API key and connectivity via model list + test completion."""
        from src.llm.client_base import ClientBase
        from openai import OpenAI

        self._refresh_config()
        service = self._config.llm_api
        model = self._config.llm
        lines = ["### LLM Connection", f"- **Service:** {service}", f"- **Model:** {model}"]

        # Step 1: Check model list / API key validity
        try:
            result = ClientBase.get_model_list(service, "GPT_SECRET_KEY.txt")
            models = result.available_models
            if models and not any("error" in m[1].lower() for m in models):
                lines.append(f"- &#x2705; Model list retrieved — {len(models)} model(s) available")
            else:
                display = models[0][0] if models else "unknown error"
                lines.append(f"- &#x274C; Connection issue: {display}")
                return "\n".join(lines)
        except Exception as e:
            lines.append(f"- &#x274C; Error listing models: {e}")
            return "\n".join(lines)

        # Step 2: Test actual completion with configured model
        try:
            api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
            if not api_key:
                lines.append(f"- &#x274C; No API key found in GPT_SECRET_KEY.txt")
                return "\n".join(lines)

            endpoints = {
                'openai': 'https://api.openai.com/v1',
                'openrouter': 'https://openrouter.ai/api/v1',
            }
            base_url = endpoints.get(service.strip().lower(), service)
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=3,
            )
            client.close()
            if response and response.choices:
                lines.append(f"- &#x2705; Test completion succeeded (model responds)")
            else:
                lines.append(f"- &#x26A0;&#xFE0F; Completion returned empty response")
        except Exception as e:
            error_str = str(e)
            if "502" in error_str or "authenticate" in error_str.lower():
                lines.append(f"- &#x274C; API auth/gateway error: {error_str[:200]}")
                lines.append(f"- **Tip:** Check your API key in GPT_SECRET_KEY.txt and verify the model `{model}` is available on {service}")
            elif "404" in error_str or "model_not_found" in error_str.lower():
                lines.append(f"- &#x274C; Model `{model}` not found on {service}")
            else:
                lines.append(f"- &#x274C; Test completion failed: {error_str[:200]}")

        return "\n".join(lines)

    def test_tts_service(self) -> str:
        """Check TTS binary / server availability."""
        self._refresh_config()
        tts = self._config.tts_service
        lines = ["### TTS Service", f"- **Selected:** {tts.display_name}"]

        if tts == TTSEnum.PIPER:
            piper_path = self._config.piper_path
            if not piper_path:
                piper_path = os.path.join(utils.resolve_path(), "piper")

            binary_name = "piper" if ON_POSIX else "piper.exe"
            binary = Path(piper_path) / binary_name
            if binary.exists():
                executable = os.access(binary, os.X_OK) if ON_POSIX else True
                if executable:
                    lines.append(f"- &#x2705; Binary found: `{binary}`")
                else:
                    lines.append(f"- &#x26A0;&#xFE0F; Binary found but not executable: `{binary}`")
            else:
                lines.append(f"- &#x274C; Binary not found: `{binary}`")

        elif tts == TTSEnum.XTTS:
            url = self._config.xtts_url
            lines.append(f"- **URL:** {url}")
            try:
                import urllib.request
                req = urllib.request.Request(f"{url}/docs", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    lines.append(f"- &#x2705; XTTS server responded (HTTP {resp.status})")
            except Exception as e:
                lines.append(f"- &#x274C; Could not reach XTTS server: {e}")

        elif tts == TTSEnum.XVASYNTH:
            xva_path = self._config.xvasynth_path
            if Path(xva_path).exists():
                lines.append(f"- &#x2705; xVASynth folder found: `{xva_path}`")
            else:
                lines.append(f"- &#x274C; xVASynth folder not found: `{xva_path}`")

            if ON_POSIX:
                lines.append(f"- &#x26A0;&#xFE0F; xVASynth cannot be auto-launched on Linux (server.exe is a Windows binary)")
                lines.append(f"- **Tip:** Start xVASynth manually via Proton/Wine, or switch to Piper/XTTS")

            # Check if xVASynth server is reachable
            try:
                import urllib.request
                req = urllib.request.Request("http://127.0.0.1:8008/", method="GET")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    lines.append(f"- &#x2705; xVASynth server is running (HTTP {resp.status})")
            except Exception:
                lines.append(f"- &#x274C; xVASynth server not reachable on port 8008")

        return "\n".join(lines)

    def test_stt_service(self) -> str:
        """Report configured STT service and check availability."""
        self._refresh_config()
        stt = self._config.stt_service
        lines = ["### STT Service", f"- **Selected:** {stt}"]

        if stt == "moonshine":
            folder = self._config.moonshine_folder
            model = self._config.moonshine_model
            lines.append(f"- **Model:** {model}")
            if folder and os.path.isdir(folder):
                lines.append(f"- &#x2705; Moonshine folder found: `{folder}`")
            else:
                lines.append(f"- &#x274C; Moonshine folder not found: `{folder}`")
        elif stt == "whisper":
            if self._config.external_whisper_service:
                url = self._config.whisper_url
                lines.append(f"- **External URL:** {url}")
                try:
                    import urllib.request
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        lines.append(f"- &#x2705; Whisper service responded (HTTP {resp.status})")
                except Exception as e:
                    lines.append(f"- &#x274C; Could not reach Whisper service: {e}")
            else:
                model = self._config.whisper_model
                device = self._config.whisper_process_device
                lines.append(f"- **Model:** {model} on {device}")
                lines.append(f"- &#x2705; Configured for local Whisper")
        else:
            lines.append(f"- Service type: {stt}")

        return "\n".join(lines)

    def test_mod_folder(self) -> str:
        """Verify mod folder path and expected subfolder structure."""
        self._refresh_config()
        mod_path = self._config.mod_path
        mod_path_base = self._config.mod_path_base
        lines = ["### Mod Folder", f"- **Base path:** `{mod_path_base}`", f"- **Voice path:** `{mod_path}`"]

        if os.path.isdir(mod_path_base):
            lines.append(f"- &#x2705; Base mod folder exists")
        else:
            lines.append(f"- &#x274C; Base mod folder not found")
            return "\n".join(lines)

        # Check expected subfolders
        expected = ["Sound", os.path.join("Sound", "Voice"), os.path.join("Sound", "Voice", "Mantella.esp")]
        for sub in expected:
            full = os.path.join(mod_path_base, sub)
            if os.path.isdir(full):
                lines.append(f"- &#x2705; `{sub}/` exists")
            else:
                lines.append(f"- &#x274C; `{sub}/` missing")

        return "\n".join(lines)

    def test_game_connection(self) -> str:
        """Report server status and port info."""
        self._refresh_config()
        port = self._config.port
        game = self._config.game
        lines = [
            "### Game Connection",
            f"- **Game:** {game.display_name if hasattr(game, 'display_name') else game}",
            f"- **HTTP Port:** {port}",
        ]

        try:
            import urllib.request
            url = f"http://127.0.0.1:{port}/ui"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                lines.append(f"- &#x2705; Mantella HTTP server is running (HTTP {resp.status})")
        except Exception:
            lines.append(f"- &#x26A0;&#xFE0F; Could not reach Mantella HTTP server on port {port} (this is normal during startup)")

        return "\n".join(lines)

    def test_microphone(self) -> str:
        """Check if a microphone input device is available and can be opened."""
        lines = ["### Microphone"]

        try:
            import sounddevice as sd

            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]

            if not input_devices:
                lines.append("- &#x274C; No input (microphone) devices found")
                return "\n".join(lines)

            default_input = sd.query_devices(kind='input')
            lines.append(f"- **Default input:** {default_input['name']}")
            lines.append(f"- **Input channels:** {default_input['max_input_channels']}")
            lines.append(f"- **Sample rate:** {default_input['default_samplerate']} Hz")

            # Try to actually open a short stream to confirm it works
            try:
                import numpy as np
                # Use the device's native sample rate to avoid paInvalidSampleRate
                native_rate = int(default_input['default_samplerate'])
                native_blocksize = int(512 * native_rate / 16000)
                stream = sd.InputStream(
                    samplerate=native_rate,
                    channels=1,
                    blocksize=native_blocksize,
                    dtype=np.float32,
                    latency='low',
                )
                stream.start()
                # Read a tiny amount of audio to confirm the stream works
                data, overflowed = stream.read(native_blocksize)
                peak = float(np.max(np.abs(data)))
                stream.stop()
                stream.close()
                lines.append(f"- &#x2705; Microphone stream opened successfully")
                if peak > 0.001:
                    lines.append(f"- &#x2705; Audio signal detected (peak: {peak:.4f})")
                else:
                    lines.append(f"- &#x26A0;&#xFE0F; Stream opened but signal is very quiet (peak: {peak:.6f}) — try speaking or check mic mute")
            except Exception as e:
                lines.append(f"- &#x274C; Could not open microphone stream: {e}")

            # List all input devices
            if len(input_devices) > 1:
                lines.append(f"- **All input devices ({len(input_devices)}):**")
                for d in input_devices[:8]:
                    lines.append(f"  - {d['name']} ({d['max_input_channels']}ch)")

        except ImportError:
            lines.append("- &#x274C; `sounddevice` module not installed")
        except Exception as e:
            lines.append(f"- &#x274C; Error querying audio devices: {e}")

        return "\n".join(lines)

    def test_stt_live(self) -> str:
        """Record a few seconds of audio and run STT transcription end-to-end."""
        self._refresh_config()
        lines = ["### STT Live Test"]

        stt_service = self._config.stt_service
        device_setting = self._config.audio_input_device
        lines.append(f"- **STT Service:** {stt_service}")
        lines.append(f"- **Audio Device:** {device_setting}")

        # Resolve the audio device
        try:
            from src.stt import Transcriber
            device_index = Transcriber._resolve_audio_device(device_setting)
        except Exception as e:
            lines.append(f"- &#x274C; Could not resolve audio device: {e}")
            return "\n".join(lines)

        # Record ~4 seconds of audio at the device's native sample rate,
        # then resample to 16 kHz for STT (avoids paInvalidSampleRate on pro
        # audio interfaces that only support 48 kHz etc.)
        TARGET_RATE = 16000
        DURATION = 4
        lines.append(f"- Recording {DURATION}s of audio...")

        try:
            import sounddevice as sd
            import math
            from scipy.signal import resample_poly as _resample_poly

            if device_index is not None:
                dev_info = sd.query_devices(device_index)
            else:
                dev_info = sd.query_devices(kind='input')
            native_rate = int(dev_info['default_samplerate'])

            audio = sd.rec(
                int(native_rate * DURATION),
                samplerate=native_rate,
                channels=1,
                dtype=np.float32,
                device=device_index,
            )
            sd.wait()
            audio = audio.flatten()

            # Resample to 16 kHz if needed
            if native_rate != TARGET_RATE:
                g = math.gcd(TARGET_RATE, native_rate)
                audio = _resample_poly(audio, TARGET_RATE // g, native_rate // g).astype(np.float32)

            peak = float(np.max(np.abs(audio)))
            lines.append(f"- **Audio peak level:** {peak:.4f}")
            if peak < 0.001:
                lines.append("- &#x26A0;&#xFE0F; Audio is very quiet — check mic mute or device selection")
        except Exception as e:
            lines.append(f"- &#x274C; Recording failed: {e}")
            return "\n".join(lines)

        # Transcribe
        try:
            if stt_service == "moonshine":
                if MoonshineOnnxModel is None:
                    lines.append("- &#x274C; `moonshine_onnx` is not installed")
                    return "\n".join(lines)

                from moonshine_onnx import load_tokenizer as _load_tok

                model_name = self._config.moonshine_model
                full_model, precision = model_name.rsplit('/', 1)
                moonshine_folder = self._config.moonshine_folder
                model_path = os.path.join(moonshine_folder, model_name)

                if os.path.exists(os.path.join(model_path, 'encoder_model.onnx')):
                    model = MoonshineOnnxModel(models_dir=model_path, model_name=full_model)
                else:
                    model = MoonshineOnnxModel(model_name=full_model, model_precision=precision)

                tokenizer = _load_tok()
                tokens = model.generate(audio[np.newaxis, :].astype(np.float32))
                text = tokenizer.decode_batch(tokens)[0]
            else:
                # Whisper (local)
                if self._config.external_whisper_service:
                    lines.append("- &#x26A0;&#xFE0F; External Whisper service — skipping local transcription test")
                    lines.append("- &#x2705; Audio recorded successfully (use the full pipeline to test external service)")
                    return "\n".join(lines)

                from faster_whisper import WhisperModel

                whisper_model = self._config.whisper_model
                process_device = self._config.whisper_process_device
                compute = "float32" if process_device != "cuda" else "default"
                model = WhisperModel(whisper_model, device=process_device, compute_type=compute)
                segments, _ = model.transcribe(audio, beam_size=5, vad_filter=False)
                text = ' '.join(seg.text for seg in segments)

            if text and text.strip():
                lines.append(f"- &#x2705; **Transcription:** {text.strip()}")
            else:
                lines.append("- &#x26A0;&#xFE0F; Transcription returned empty — try speaking louder or check the model")
        except Exception as e:
            lines.append(f"- &#x274C; Transcription failed: {e}")

        return "\n".join(lines)

    def test_all(self) -> str:
        """Run all checks, return combined markdown."""
        self._refresh_config()
        sections = [
            self.test_platform_info(),
            self.test_llm_connection(),
            self.test_tts_service(),
            self.test_stt_service(),
            self.test_microphone(),
            self.test_mod_folder(),
            self.test_game_connection(),
        ]
        return "\n\n---\n\n".join(sections)
