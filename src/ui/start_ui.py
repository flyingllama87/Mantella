from fastapi import FastAPI
from fastapi.responses import FileResponse
import webbrowser
import gradio as gr
from src.config.config_loader import ConfigLoader
from src.http.routes.routeable import routeable
from src.ui.settings_ui_constructor import SettingsUIConstructor
from src.ui.diagnostics import DiagnosticsRunner
import src.utils as utils

logger = utils.get_logger()


class StartUI(routeable):
    BANNER = "docs/_static/img/mantella_banner.png"
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config, False)
        self.__constructor = SettingsUIConstructor()
        self.__diagnostics = DiagnosticsRunner(config)

    def create_main_block(self) -> gr.Blocks:
        with gr.Blocks(title="Mantella", fill_height=True, analytics_enabled=False, theme= self.__get_theme(), css=self.__load_css()) as main_block:
            # with gr.Tab("Settings") as tabs:
            settings_page = self.__generate_settings_page()
            # with gr.Tab("Chat with NPCs", interactive=False):
            #     self.__generate_chat_page()
            # with gr.Tab("NPC editor", interactive=False):
            #     self.__generate_character_editor_page()
            self.__generate_diagnostics_page()

            with gr.Row(elem_classes="custom-footer"):
                gr.HTML("""
                    <div class="custom-footer">
                        <a href="https://art-from-the-machine.github.io/Mantella/" target="_blank">Mantella Installation Guide</a>
                    </div>
                """)
        return main_block

    def __generate_settings_page(self) -> gr.Column:
        # with gr.Column() as settings:
        for cf in self._config.definitions.base_groups:
            if not cf.is_hidden:
                with gr.Tab(cf.name):
                    cf.accept_visitor(self.__constructor)
        return None #settings
    
    def __generate_chat_page(self):
        return gr.Column()
    
    def __generate_character_editor_page(self):
        return gr.Column()

    def __generate_diagnostics_page(self):
        with gr.Tab("Diagnostics"):
            output = gr.Markdown(value="Click a button below to run diagnostics.")
            with gr.Row():
                btn_all = gr.Button("Test All", variant="primary")
            with gr.Row():
                btn_llm = gr.Button("LLM")
                btn_tts = gr.Button("TTS")
                btn_stt = gr.Button("STT")
                btn_mic = gr.Button("Microphone")
                btn_mod = gr.Button("Mod Folder")
                btn_game = gr.Button("Game")
            with gr.Row():
                btn_stt_live = gr.Button("Test STT (Live)", variant="secondary")
                gr.Markdown("*Records ~4s of audio, then transcribes it with your configured STT service.*")

            btn_all.click(fn=self.__diagnostics.test_all, outputs=output)
            btn_llm.click(fn=self.__diagnostics.test_llm_connection, outputs=output)
            btn_tts.click(fn=self.__diagnostics.test_tts_service, outputs=output)
            btn_stt.click(fn=self.__diagnostics.test_stt_service, outputs=output)
            btn_mic.click(fn=self.__diagnostics.test_microphone, outputs=output)
            btn_mod.click(fn=self.__diagnostics.test_mod_folder, outputs=output)
            btn_game.click(fn=self.__diagnostics.test_game_connection, outputs=output)
            btn_stt_live.click(fn=self.__diagnostics.test_stt_live, outputs=output)

    def __get_theme(self):
        return gr.themes.Soft(primary_hue="green",
                            secondary_hue="green",
                            neutral_hue="zinc",
                            font=['Montserrat', 'ui-sans-serif', 'system-ui', 'sans-serif'],
                            font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']).set(
                                input_text_size='*text_md',
                                input_padding='*spacing_md',
                            )

    
    def add_route_to_server(self, app: FastAPI):
        @app.get("/favicon.ico")
        async def favicon():
            return FileResponse("Mantella.ico")

        gr.mount_gradio_app(app,
                            self.create_main_block(),
                            path="/ui")
        
        link = f'http://localhost:{str(self._config.port)}/ui?__theme=dark'
        logger.log(24, f'\nMantella settings can be changed via this link:')
        logger.log(25, link)
        if self._config.auto_launch_ui == True:
            if not webbrowser.open(link, new=2):
                logger.warning('\nFailed to open Mantella settings UI automatically. To edit settings, see here:')
                logger.log(25, link)
    
    def __load_css(self):
        with open('src/ui/style.css', 'r') as file:
            css_content = file.read()
        return css_content
    
    def _setup_route(self):
        pass

    