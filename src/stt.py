import sys
import math
import numpy as np
from faster_whisper import WhisperModel
from src.config.config_loader import ConfigLoader
import src.utils as utils
import requests
import json
import io
from pathlib import Path
from openai import OpenAI
from typing import Optional
from datetime import datetime
import queue
import threading
import time
import os
import wave
from scipy.signal import resample_poly
try:
    from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
except ImportError:
    MoonshineOnnxModel = None
    load_tokenizer = None
import onnxruntime as ort
from scipy.io import wavfile
from sounddevice import InputStream
from silero_vad_lite import SileroVAD

logger = utils.get_logger()


import onnxruntime as ort
ort.set_default_logger_severity(4)


class Transcriber:
    """Handles real-time speech-to-text transcription using Moonshine."""
    
    SAMPLING_RATE = 16000
    CHUNK_SIZE = 512  # Required chunk size for Silero VAD
    CHUNK_DURATION = CHUNK_SIZE / SAMPLING_RATE  # Explicit calculation of chunk duration in seconds
    LOOKBACK_CHUNKS = 5  # Number of chunks to keep in buffer when not recording
    
    @utils.time_it
    def __init__(self, config: ConfigLoader, stt_secret_key_file: str, secret_key_file: str):
        self.loglevel = 27
        logger.log(27, f"STT: Initializing Transcriber (service={config.stt_service}, language={config.stt_language})")
        self.language = config.stt_language
        self.task = "translate" if config.stt_translate == 1 else "transcribe"
        self.stt_service = config.stt_service
        self.full_moonshine_model = config.moonshine_model
        self.moonshine_model, self.moonshine_precision = self.full_moonshine_model.rsplit('/', 1)
        self.moonshine_folder = config.moonshine_folder
        self.moonshine_model_path = os.path.join(self.moonshine_folder, self.full_moonshine_model)
        self.whisper_model = config.whisper_model
        self.process_device = config.whisper_process_device
        self.listen_timeout = config.listen_timeout
        self.external_whisper_service = config.external_whisper_service
        self.whisper_service = config.whisper_url
        self.whisper_url = self.__get_endpoint(config.whisper_url)
        self.prompt = ''
        self.show_mic_warning = True
        self.play_cough_sound = config.play_cough_sound
        self.transcription_times = []
        self.proactive_mic_mode = config.proactive_mic_mode
        self.min_refresh_secs = config.min_refresh_secs # Minimum time between transcription updates
        self.refresh_freq = self.min_refresh_secs // self.CHUNK_DURATION # Number of chunks between transcription updates
        self.pause_threshold = config.pause_threshold
        self._temporary_pause_override: float | None = None  # Temporary pause threshold for Listen action
        self.audio_threshold = config.audio_threshold
        self.audio_input_device = self._resolve_audio_device(config.audio_input_device)
        self._update_native_rate()

        logger.log(self.loglevel, f"Audio threshold set to {self.audio_threshold}. If the mic is not picking up your voice, try lowering this `Speech-to-Text`->`Audio Threshold` value in the Mantella UI. If the mic is picking up too much background noise, try increasing this value.\n")

        self.__audio_input_error_count = 0
        self.__mic_input_process_error_count = 0
        self.__processing_audio_error_count = 0
        self.__warning_frequency = 5
        
        self.__save_mic_input = config.save_mic_input
        if self.__save_mic_input:
            self.__mic_input_path: str = os.path.join(config.save_folder, 'data', 'tmp', 'mic')
            os.makedirs(self.__mic_input_path, exist_ok=True)

        self.__stt_secret_key_file = stt_secret_key_file
        self.__secret_key_file = secret_key_file
        self.__api_key: str | None = self.__get_api_key()
        self.__initial_client: OpenAI | None = None
        if (self.stt_service == 'whisper') and (self.__api_key) and ('openai' in self.whisper_url) and (self.external_whisper_service):
            self.__initial_client = self.__generate_sync_client() # initialize first client in advance to save time

        self.__ignore_list = ['', 'thank you', 'thank you for watching', 'thanks for watching', 'the transcript is from the', 'the', 'thank you very much', "thank you for watching and i'll see you in the next video", "we'll see you in the next video", 'see you next time', 'you']
        
        self.transcribe_model: WhisperModel | MoonshineOnnxModel | None = None
        if self.stt_service == 'whisper':
            # if using faster_whisper, load model selected by player, otherwise skip this step
            if not self.external_whisper_service:
                if self.process_device == 'cuda':
                    logger.error(f'''Depending on your NVIDIA CUDA version, setting the Whisper process device to `cuda` may cause errors! For more information, see here: https://github.com/SYSTRAN/faster-whisper#gpu''')
                    try:
                        self.transcribe_model = WhisperModel(self.whisper_model, device=self.process_device)
                    except Exception as e:
                        utils.play_error_sound()
                        raise e
                else:
                    self.transcribe_model = WhisperModel(self.whisper_model, device=self.process_device, compute_type="float32")
        else:
            if MoonshineOnnxModel is None:
                raise ImportError("Moonshine STT service is selected but 'moonshine_onnx' is not installed. Install it with: pip install moonshine-onnx")

            if self.language != 'en':
                logger.warning(f"Selected language is '{self.language}', but Moonshine only supports English. Please change the selected speech-to-text model to Whisper in `Speech-to-Text`->`STT Service` in the Mantella UI")

            if self.moonshine_model == 'moonshine/tiny':
                logger.warning('Speech-to-text model set to Moonshine Tiny. If mic input is being transcribed incorrectly, try switching to a larger model in the `Speech-to-Text` tab of the Mantella UI')
            
            if os.path.exists(f'{self.moonshine_model_path}/encoder_model.onnx'):
                logger.log(self.loglevel, 'Loading local Moonshine model...')
                self.transcribe_model = MoonshineOnnxModel(models_dir=self.moonshine_model_path, model_name=self.moonshine_model)
            else:
                logger.log(self.loglevel, 'Loading Moonshine model from Hugging Face...')
                self.transcribe_model = MoonshineOnnxModel(model_name=self.moonshine_model, model_precision=self.moonshine_precision)
            self.tokenizer = load_tokenizer()
        
        # Initialize VAD
        self.vad = SileroVAD(self.SAMPLING_RATE)
        
        # Audio processing state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._audio_queue = queue.Queue()
        self._stream: Optional[InputStream] = None
        
        # Threading and synchronization
        self._lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Speech detection state
        self._speech_detected = False
        self._speech_end_time = 0
        self._last_update_time = 0
        self._current_transcription = ""
        self._transcription_ready = threading.Event()
        self._consecutive_empty_count = 0
        self._max_consecutive_empty = 10

    @staticmethod
    def _resolve_audio_device(device_setting: str) -> int | None:
        """Resolve the audio device setting to a sounddevice device index.

        Args:
            device_setting: 'Default', a device index string, a dropdown value like
                            '2: USB Mic (2ch)', or a substring of a device name.

        Returns:
            Device index (int) or None for system default.
        """
        import sounddevice as sd
        import re

        if not device_setting or device_setting.strip().lower() == 'default':
            return None

        value = device_setting.strip()

        # Try dropdown format "N: Device Name (Xch)"
        m = re.match(r'^(\d+):\s', value)
        if m:
            idx = int(m.group(1))
            try:
                device_info = sd.query_devices(idx)
                if device_info['max_input_channels'] > 0:
                    logger.log(27, f"Audio input device set to #{idx}: {device_info['name']}")
                    return idx
            except sd.PortAudioError:
                pass

        # Try as plain integer index
        try:
            idx = int(value)
            device_info = sd.query_devices(idx)
            if device_info['max_input_channels'] > 0:
                logger.log(27, f"Audio input device set to #{idx}: {device_info['name']}")
                return idx
        except (ValueError, sd.PortAudioError):
            pass

        # Try as substring match on device name
        devices = sd.query_devices()
        # Strip dropdown formatting ("N: Device Name (Xch)" -> "Device Name")
        # so the substring match works even when device indices have shifted.
        name_for_match = re.sub(r'^\d+:\s*', '', value)
        name_for_match = re.sub(r'\s*\(\d+ch\)\s*$', '', name_for_match)
        needle = name_for_match.lower()
        for i, d in enumerate(devices):
            if needle in d['name'].lower() and d['max_input_channels'] > 0:
                logger.log(27, f"Audio input device matched '{device_setting}' -> #{i}: {d['name']}")
                return i

        logger.warning(f"Could not find audio input device '{device_setting}', using system default")
        return None

    def _update_native_rate(self) -> None:
        """Query the audio device's native sample rate and compute resampling parameters."""
        import sounddevice as sd
        try:
            if self.audio_input_device is not None:
                dev_info = sd.query_devices(self.audio_input_device)
            else:
                try:
                    dev_info = sd.query_devices(kind='input')
                except Exception:
                    # PortAudio has no default input device (-1); find first input device
                    for i, d in enumerate(sd.query_devices()):
                        if d['max_input_channels'] > 0:
                            dev_info = d
                            self.audio_input_device = i
                            logger.log(self.loglevel, f"No default input device; using #{i}: {d['name']}")
                            break
                    else:
                        raise RuntimeError("No input audio devices found")
            self._native_rate = int(dev_info['default_samplerate'])
        except Exception:
            self._native_rate = self.SAMPLING_RATE  # fallback to 16 kHz

        if self._native_rate != self.SAMPLING_RATE:
            self._native_blocksize = int(self.CHUNK_SIZE * self._native_rate / self.SAMPLING_RATE)
            g = math.gcd(self.SAMPLING_RATE, self._native_rate)
            self._resample_up = self.SAMPLING_RATE // g
            self._resample_down = self._native_rate // g
            logger.log(self.loglevel, f"Audio device native rate is {self._native_rate} Hz; will resample to {self.SAMPLING_RATE} Hz (ratio {self._resample_down}:{self._resample_up})")
        else:
            self._native_blocksize = self.CHUNK_SIZE
            self._resample_up = 1
            self._resample_down = 1

    @property
    def is_listening(self) -> bool:
        """Returns True if actively listening."""
        return self._processing_thread is not None and self._processing_thread.is_alive()

    @property
    def has_player_spoken(self) -> bool:
        """Check if speech has been detected."""
        with self._lock:
            return self._speech_detected
    
    def set_temporary_pause(self, pause_seconds: float) -> None:
        """Set a temporary pause threshold override for the next transcription
        
        This is used by the Listen action to give the player more time to formulate their response.
        The temporary pause will be automatically cleared after the next successful transcription.
        
        Args:
            pause_seconds: The pause threshold in seconds
        """
        with self._lock:
            self._temporary_pause_override = pause_seconds
            # If already listening and speech hasn't been detected yet, recreate VAD iterator with new pause threshold
            if self._running and not self._speech_detected:
                self.vad_iterator = self._create_vad_iterator()
        

    @utils.time_it
    def __generate_sync_client(self):
        if self.__initial_client:
            client = self.__initial_client
            self.__initial_client = None # do not reuse the same client
        else:
            client = OpenAI(api_key=self.__api_key, base_url=self.whisper_url)

        return client
    

    @utils.time_it
    def __get_endpoint(self, whisper_url):
        known_endpoints = {
            'OpenAI': 'https://api.openai.com/v1',
            'Groq': 'https://api.groq.com/openai/v1',
            'whisper.cpp': 'http://127.0.0.1:8080/inference',
        }
        if whisper_url in known_endpoints:
            return known_endpoints[whisper_url]
        else: # if not found, use value as is
            return whisper_url
        

    @utils.time_it
    def __get_api_key(self) -> str:
        if self.external_whisper_service:
            try: # first check mod folder for stt secret key
                mod_parent_folder = str(Path(utils.resolve_path()).parent.parent.parent)
                with open(os.path.join(mod_parent_folder, self.__stt_secret_key_file), 'r') as f:
                    api_key: str = f.readline().strip()
            except: # check locally (same folder as exe) for stt secret key
                try:
                    with open(self.__stt_secret_key_file, 'r') as f:
                        api_key: str = f.readline().strip()
                except:
                    try: # first check mod folder for secret key
                        mod_parent_folder = str(Path(utils.resolve_path()).parent.parent.parent)
                        with open(os.path.join(mod_parent_folder, self.__secret_key_file), 'r') as f:
                            api_key: str = f.readline().strip()
                    except: # check locally (same folder as exe) for secret key
                        with open(self.__secret_key_file, 'r') as f:
                            api_key: str = f.readline().strip()
                
            if not api_key:
                logger.error(f'''No secret key found in GPT_SECRET_KEY.txt. Please create a secret key and paste it in your Mantella mod folder's GPT_SECRET_KEY.txt file.
If using OpenAI, see here on how to create a secret key: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key
If you would prefer to run speech-to-text locally, please ensure the `Speech-to-Text`->`External Whisper Service` setting in the Mantella UI is disabled.''')
                input("Press Enter to continue.")
                sys.exit(0)
            return api_key


    @utils.time_it
    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Moonshine model."""
        # Count speech end time from when the last transcribe is called
        self._speech_end_time = time.time()
        if self.stt_service == 'moonshine':
            transcription = self.moonshine_transcribe(audio)
        else:
            transcription = self.whisper_transcribe(audio, self.prompt)

        self.transcription_times.append((time.time() - self._speech_end_time))
        if (self.proactive_mic_mode) and (len(self.transcription_times) % 5 == 0):
            max_transcription_time = max(self.transcription_times[-5:])
            if max_transcription_time > self.min_refresh_secs:
                logger.warning(f'Mic transcription took {round(max_transcription_time,3)} to process. To improve performance, try setting `Speech-to-Text`->`Refresh Frequency` to a value slightly higher than {round(max_transcription_time,3)} in the Mantella UI')

        if self.proactive_mic_mode:
            logger.log(self.loglevel, f'Interim transcription: {transcription}')
        
        # Filter prompt-echo hallucinations (Whisper repeating its initial_prompt
        # back when it receives silence or short noise bursts)
        if transcription and self._is_hallucination(transcription):
            logger.debug(f"STT: Filtered hallucination: '{transcription.strip()}'")
            transcription = ''

        # Only update the transcription if it contains a value, otherwise keep the existing transcription
        if transcription:
            return transcription
        else:
            self._consecutive_empty_count += 1
            return self._current_transcription


    def _is_hallucination(self, text: str) -> bool:
        """Detect Whisper hallucinations (prompt echoes, repetitive phrases).

        Whisper tends to echo back its initial_prompt when given silence or
        very short noise bursts.  It also sometimes produces repetitive
        phrases like 'Thank you. Thank you. Thank you.'
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return False

        # Prompt-echo: the prompt is "This is a conversation with <NPC> in <location>."
        # Whisper hallucinates this exact text (or repetitions of it) from noise.
        if 'this is a conversation with' in cleaned:
            return True

        # Repetitive hallucination: same sentence repeated multiple times
        sentences = [s.strip() for s in cleaned.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) >= 2 and len(set(sentences)) == 1:
            return True

        return False


    @utils.time_it
    def whisper_transcribe(self, audio: np.ndarray, prompt: str):
        if self.transcribe_model: # local model
            segments, _ = self.transcribe_model.transcribe(audio, task=self.task, language=self.language, beam_size=5, vad_filter=False, initial_prompt=prompt)
            result_text = ' '.join(segment.text for segment in segments)
            if utils.clean_text(result_text) in self.__ignore_list: # common phrases hallucinated by Whisper
                return ''
            return result_text
        
        # Server versions of Whisper require the audio data to be a file type
        audio_file = io.BytesIO()
        wavfile.write(audio_file, self.SAMPLING_RATE, audio)
        # Audio file needs a name or else Whisper gets angry
        audio_file.name = 'out.wav'

        if 'openai' in self.whisper_url: # OpenAI compatible endpoint
            client = self.__generate_sync_client()
            try:
                response_data = client.audio.transcriptions.create(model=self.whisper_model, language=self.language, file=audio_file, prompt=prompt)
            except Exception as e:
                utils.play_error_sound()
                if e.code in [404, 'model_not_found']:
                    if self.whisper_service == 'OpenAI':
                        logger.error(f"Selected Whisper model '{self.whisper_model}' does not exist in the OpenAI service. Try changing 'Speech-to-Text'->'Model Size' to 'whisper-1' in the Mantella UI")
                    elif self.whisper_service == 'Groq':
                        logger.error(f"Selected Whisper model '{self.whisper_model}' does not exist in the Groq service. Try changing 'Speech-to-Text'->'Model Size' to one of the following models in the Mantella UI: https://console.groq.com/docs/speech-text#supported-models")
                    else:
                        logger.error(f"Selected Whisper model '{self.whisper_model}' does not exist in the selected service {self.whisper_service}. Try changing 'Speech-to-Text'->'Model Size' to a compatible model in the Mantella UI")
                else:
                    logger.error(f'STT error: {e}')
                input("Press Enter to exit.")
            client.close()
            if utils.clean_text(response_data.text) in self.__ignore_list: # common phrases hallucinated by Whisper
                return ''
            return response_data.text.strip()
        else: # custom server model
            data = {'model': self.whisper_model, 'prompt': prompt}
            files = {'file': ('audio.wav', audio_file, 'audio/wav')}
            response = requests.post(self.whisper_url, files=files, data=data)
            if response.status_code != 200:
                logger.error(f'STT Error: {response.content}')
            response_data = json.loads(response.text)
            if 'text' in response_data:
                if utils.clean_text(response_data['text']) in self.__ignore_list: # common phrases hallucinated by Whisper
                    return ''
                return response_data['text'].strip()
            

    @utils.time_it
    def moonshine_transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Moonshine model"""
        tokens = self.transcribe_model.generate(audio[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]
        text = self.ensure_sentence_ending(text)
        
        return text
    

    def ensure_sentence_ending(self, text: str) -> str:
        '''Moonshine transcriptions tend to be missing sentence-ending characters, which can confuse LLMs'''
        if not text:  # Handle empty string
            return text
        
        end_chars = {'.', '?', '!', ':', ';', '。'}
        
        if text[-1] == ',':
            return text[:-1] + '.'
        elif text[-1] not in end_chars:
            return text + '.'
        
        return text


    @utils.time_it
    def start_listening(self, prompt: str = '') -> None:
        '''Start background listening thread'''
        if self._running:
            return
            
        self._running = True
        self._reset_state()
        self.prompt = prompt
        
        # Start audio stream at the device's native sample rate.
        # On Linux, opening an ALSA device by index can fail when
        # PipeWire/PulseAudio already owns it, so fall back to the
        # system default device (which routes through PipeWire/Pulse).
        try:
            self._stream = InputStream(
                device=self.audio_input_device,
                samplerate=self._native_rate,
                channels=1,
                blocksize=self._native_blocksize,
                dtype=np.float32,
                callback=self._create_input_callback(self._audio_queue),
            )
        except Exception as e:
            if self.audio_input_device is not None:
                logger.warning(f"Could not open audio device #{self.audio_input_device} at {self._native_rate} Hz: {e}. Falling back to system default device.")
                self.audio_input_device = None
                self._update_native_rate()
                self._stream = InputStream(
                    device=None,
                    samplerate=self._native_rate,
                    channels=1,
                    blocksize=self._native_blocksize,
                    dtype=np.float32,
                    callback=self._create_input_callback(self._audio_queue),
                )
            else:
                raise
        self._stream.start()
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_audio,
            daemon = True
        )
        self._processing_thread.start()
        logger.log(self.loglevel, 'Listening...')


    def _process_audio(self) -> None:
        """Process audio data in a separate thread."""
        lookback_size = self.LOOKBACK_CHUNKS * self.CHUNK_SIZE
        chunk_count = 0
        
        while self._running:
            try:
                # Get audio chunk and status from queue
                chunk, status = self._audio_queue.get(timeout=0.1)
                if status:
                    if self.__processing_audio_error_count % self.__warning_frequency == 0:
                        logger.debug(f"STT: audio stream status: {status}")
                    self.__processing_audio_error_count += 1
                    # Don't skip the chunk — "input overflow" means some
                    # prior audio was lost, but this chunk's data is still
                    # valid.  Discarding it creates gaps that confuse VAD.

                # Resample from native rate to 16 kHz if needed
                if self._native_rate != self.SAMPLING_RATE:
                    chunk = resample_poly(chunk, self._resample_up, self._resample_down).astype(np.float32)
                    # For non-integer ratios (e.g. 44100→16000) the output
                    # may be ±1 sample off; Silero VAD needs exactly CHUNK_SIZE.
                    if len(chunk) < self.CHUNK_SIZE:
                        chunk = np.pad(chunk, (0, self.CHUNK_SIZE - len(chunk)))
                    elif len(chunk) > self.CHUNK_SIZE:
                        chunk = chunk[:self.CHUNK_SIZE]

                with self._lock:
                    # Update audio buffer
                    self._audio_buffer = np.concatenate((self._audio_buffer, chunk))
                    if not self._speech_detected:
                        # Keep limited lookback buffer when not recording
                        self._audio_buffer = self._audio_buffer[-lookback_size:]

                    # Process with VAD
                    probability = self.vad.process(chunk)

                    # Handle speech detection
                    if probability > self.audio_threshold:
                        self._last_update_time = time.time()

                    if probability > self.audio_threshold and not self._speech_detected:
                        logger.log(self.loglevel, 'Speech detected')
                        self._speech_detected = True

                    if probability <= self.audio_threshold and self._speech_detected and time.time() - self._last_update_time > self.pause_threshold:
                        logger.log(self.loglevel, 'Speech ended')
                        # If proactive mode is disabled, transcribe mic input only when speech end has been detected
                        if not self.proactive_mic_mode:
                            self._current_transcription = self._transcribe(self._audio_buffer)
                        if self.__save_mic_input:
                            self._save_audio(self._audio_buffer)

                        self._transcription_ready.set()
                        self._reset_state()

                    # Update transcription periodically during speech
                    elif self._speech_detected:
                        chunk_count += 1

                        # Check for maximum speech duration
                        if (len(self._audio_buffer) / self.SAMPLING_RATE) > self.listen_timeout:
                            logger.warning(f'Listen timeout of {self.listen_timeout} seconds reached. Processing mic input...')
                            self._current_transcription = self._transcribe(self._audio_buffer)
                            self._transcription_ready.set()

                            self._reset_state()
                        # Regular update during speech
                        elif (self.proactive_mic_mode) and (chunk_count >= self.refresh_freq):
                            logger.debug(f'Transcribing {self.min_refresh_secs} of mic input...')
                            self._current_transcription = self._transcribe(self._audio_buffer)

                            if self._consecutive_empty_count >= self._max_consecutive_empty:
                                logger.warning(f'Could not transcribe input')
                                self._transcription_ready.set()
                                self._reset_state()

                            chunk_count = 0  # Reset counter
            
            except queue.Empty:
                logger.debug('Queue is empty')
                continue
            except Exception as e:
                if self.__mic_input_process_error_count % self.__warning_frequency == 0:
                    logger.log(23, f'STT WARNING: Error processing mic input: {str(e)}')
                self.__mic_input_process_error_count += 1
                self._reset_state()
                time.sleep(0.1)


    def _create_input_callback(self, q: queue.Queue):
        """Create callback for audio input stream."""
        def input_callback(indata, frames, time, status):
            if status:
                if self.__audio_input_error_count % self.__warning_frequency == 0:
                    logger.debug(f"STT: audio callback status: {status}")
                self.__audio_input_error_count += 1
            # Store both data and status in queue
            q.put((indata.copy().flatten(), status))
        return input_callback


    def _reset_state(self) -> None:
        """Reset internal state."""
        self._speech_detected = False
        self._audio_buffer = np.array([], dtype=np.float32)
        self.vad = SileroVAD(self.SAMPLING_RATE)
        self._consecutive_empty_count = 0


    @utils.time_it
    def _save_audio(self, audio: np.ndarray) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(self.__mic_input_path, f'mic_input_{timestamp}.wav')
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.SAMPLING_RATE)
            # Convert float32 to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())


    @utils.time_it
    def get_latest_transcription(self, silence_timeout: float = 0) -> str | None:
        """Get the latest transcription, blocking until speech ends or silence timeout
        
        Args:
            silence_timeout: How long to wait (in seconds) for speech before returning None. If 0 or negative, waits indefinitely.
        
        Returns:
            The transcribed text, or None if silence_timeout elapsed without any speech being detected
        """
        while True:
            # Use timeout only if silence_timeout > 0 and player hasn't started speaking yet
            use_timeout = silence_timeout > 0 and not self._speech_detected
            timeout_value = silence_timeout if use_timeout else None
            
            received_transcription = self._transcription_ready.wait(timeout=timeout_value)
            
            if not received_transcription and use_timeout and not self._speech_detected:
                logger.log(self.loglevel, f"Silence timeout of {silence_timeout} seconds reached without speech")
                return None
            
            with self._lock:
                transcription = self._current_transcription
                self._current_transcription = ''
                if transcription:
                    self._transcription_ready.clear()
                    self._speech_detected = False
                    self._temporary_pause_override = None  # Reset temporary pause after transcription
                    logger.log(self.loglevel, f"Player said '{transcription.strip()}'")
                    return transcription
                
            if self.play_cough_sound:
                utils.play_no_mic_input_detected_sound()
            logger.warning('Could not detect speech from mic input')

            self._transcription_ready.clear()
            self._speech_detected = False
            self._current_transcription = ''

            time.sleep(0.1)


    def stop_listening(self) -> None:
        """Stop listening for speech."""
        if not self._running:
            return
            
        self._running = False
        self._speech_detected = False
        
        # Stop and clean up audio stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Wait for processing thread to finish
        if self._processing_thread:
            self._processing_thread.join()  # timeout=1.0 Add timeout to prevent hanging
            self._processing_thread = None
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self._reset_state()
        logger.log(self.loglevel, 'Stopped listening for mic input')


    @staticmethod
    @utils.time_it
    def activation_name_exists(transcript: str, activation_names: str | list[str]) -> bool:
        """Identifies keyword in the input transcript"""
        if not transcript:
            return False

        # Convert to a list even if there is only one activation name
        if isinstance(activation_names, str):
            activation_names = [activation_names]

        # Check for a match among individual words in the transcript
        transcript_words = transcript.split()
        if set(transcript_words).intersection(activation_names):
            return True

        # Alternatively, if the entire transcript is a keyword, return True
        for activation_name in activation_names:
            if transcript == activation_name:
                return True

        return False


    @staticmethod
    @utils.time_it
    def _remove_activation_word(transcript, activation_name):
        transcript = transcript.replace(activation_name, '')
        return transcript