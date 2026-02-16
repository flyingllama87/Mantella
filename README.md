# Mantella <a href="https://www.nexusmods.com/skyrimspecialedition/mods/98631" target="_blank" style="padding-right: 8px;"><img src="./img/nexus_mods_link.png" alt="Mantella Skyrim Nexus Mods link" width="auto" height="28"/></a><a href="https://www.nexusmods.com/fallout4/mods/79747" target="_blank"><img src="./img/nexus_mods_fallout4_link.png" alt="Mantella Fallout 4 Nexus Mods link" width="auto" height="28"/></a>

<img src="./img/mantella_logo_github.png" align="right" alt="Mantella logo" width="150" height="auto">

> Bring Skyrim and Fallout 4 NPCs to life with AI

Mantella is a Skyrim and Fallout 4 mod which allows you to naturally speak to NPCs using speech-to-text ([Moonshine](https://github.com/usefulsensors/moonshine) / [Whisper](https://github.com/openai/whisper)), LLMs, and text-to-speech ([Piper](https://github.com/rhasspy/piper) / [xVASynth](https://github.com/DanRuta/xVA-Synth) / [XTTS](https://www.nexusmods.com/skyrimspecialedition/mods/113445)).  

Click below or [here](https://youtu.be/FLmbd48r2Wo?si=QLe2_E1CogpxlaS1) to see the full trailer:

<a href="https://youtu.be/FLmbd48r2Wo?si=QLe2_E1CogpxlaS1
" target="_blank"><img src="./img/mantella_trailer.gif"
alt="Mantella trailer link" width="auto" height="220"/></a>

For more details, see [here](https://art-from-the-machine.github.io/Mantella/index.html).

# Linux update and disclaimer.

This version of Mantella adds a variety of quality of life improvements and improves linux support. LLMs were used to make these changes and I can only attest that they worked for my setup. This fork will not be maintained. Below is the tested component stack:
- Python 3.11 venv
- Ubuntu 24.04
- Whisper for STT
- xVASynth for TTS
- Skyrim Special Edition from steam, played on proton 9.0
- Mod Organizer 2
- xVASynth from steam

Best results with push to speak (under Mantella settings).

### Major Update Summary

-   **Broadened Cross-Platform Support:** Refactored file paths and removed hardcoded Windows dependencies to ensure smooth operation on Linux and other POSIX systems. This includes better handling for case-sensitive filesystems, alternative screenshot methods, and OS-specific executables.
-   **Enhanced Audio Compatibility:** Added a UI selector for audio input devices and implemented native sample rate resampling. This resolves common errors with professional audio interfaces and improves resilience against audio stream overflows or missing default devices.
-   **Smarter Speech-to-Text (STT):** Significantly improved transcription accuracy by actively filtering out Whisper model hallucinations (like prompt echoes) and ignoring background NPC game audio picked up by the microphone.
-   **Increased Stability & Auto-Recovery:** Hardened the system against hangs and crashes. The application now gracefully handles missing voice models, timeouts from unresponsive game actions, unsupported LLM parameters, and ensures clean process shutdowns.
-   **New Diagnostics & Advanced LLM Controls:** Introduced a comprehensive Diagnostics module to test core services (LLM, TTS, STT, and mic). Added a "Reasoning Effort" toggle for advanced models (like the OpenAI o-series or DeepSeek R1) and expanded console logging for better lifecycle and error tracking.




# Attributions
Mantella uses material from the "[Skyrim: Characters](https://elderscrolls.fandom.com/wiki/Category:Skyrim:_Characters)" articles on the [Elder Scrolls wiki](https://elderscrolls.fandom.com/wiki/The_Elder_Scrolls_Wiki) at [Fandom](https://www.fandom.com/) and is licensed under the [Creative Commons Attribution-Share Alike License](https://creativecommons.org/licenses/by-sa/3.0/).
