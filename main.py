from src.http.http_server import http_server
import os
import signal
import threading
import traceback
from src.http.routes.routeable import routeable
from src.http.routes.mantella_route import mantella_route
from src.setup import MantellaSetup
from src.ui.start_ui import StartUI
import src.utils as utils


def _force_exit():
    """Force-kill the process when graceful shutdown hangs (e.g. PortAudio threads)."""
    os._exit(1)


def _make_signal_handler(logger):
    """Create a signal handler with a force-exit fallback.

    On first signal: log, start a 3-second force-exit timer, then re-raise
    the signal with the default handler so uvicorn can attempt graceful shutdown.
    On second signal: force-exit immediately.
    """
    _already_called = False

    def handler(sig, frame):
        nonlocal _already_called
        if _already_called:
            # Second signal → force exit immediately
            os._exit(1)
        _already_called = True
        logger.info(f"Shutting down...")
        # Give uvicorn / threads 3 seconds to shut down, then force exit
        timer = threading.Timer(3.0, _force_exit)
        timer.daemon = True
        timer.start()
        # Re-raise with default handler so uvicorn can do graceful shutdown
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)

    return handler


def main():
    try:
        mantella_version = '0.14 Preview 1'
        config, language_info = MantellaSetup().initialise(
            config_file='config.ini',
            logging_file='logging.log',
            language_file='data/language_support.csv',
            mantella_version=mantella_version)

        logger = utils.get_logger()

        # Register signal handlers so Ctrl+C / kill work even when
        # PortAudio callback threads or blocking STT calls prevent
        # Python from handling KeyboardInterrupt normally.
        _handler = _make_signal_handler(logger)
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

        logger.log(24, f'\nMantella v{mantella_version}')

        mantella_http_server = http_server()

        should_debug_http = config.show_http_debug_messages
        conversation = mantella_route(
            config=config,
            stt_secret_key_file='STT_SECRET_KEY.txt',
            image_secret_key_file='IMAGE_SECRET_KEY.txt',
            function_llm_secret_key_file='FUNCTION_GPT_SECRET_KEY.txt',
            secret_key_file='GPT_SECRET_KEY.txt',
            language_info=language_info,
            show_debug_messages=should_debug_http
        )
        ui = StartUI(config)
        routes: list[routeable] = [conversation, ui]

        mantella_http_server.start(int(config.port), routes, config.play_startup_sound, should_debug_http)

    except Exception as e:
        logger.error("".join(traceback.format_exception(e)))
        input("Press Enter to exit.")

if __name__ == '__main__':
    main()
