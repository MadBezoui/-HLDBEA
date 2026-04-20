import os
import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, RichLog, Collapsible, OptionList
from textual.containers import Vertical, Horizontal, Grid
from textual.worker import Worker, WorkerState

class ExperimentLauncher(App):
    CSS = """
    Screen { align: center middle; }
    #main_grid { grid-size: 2; grid-columns: 1fr 2fr; padding: 1; }
    #config_panel { border: thick $primary; padding: 1; height: 100%;  overflow-y: auto; }
    #log_panel { border: solid $accent; background: $surface; height: 100%; }
    
    Label { margin-top: 1; text-style: bold; color: $text; }
    ListView { height: auto; border: solid $primary-darken-2; margin-bottom: 1; }
    Button { width: 100%; margin-top: 1; }
    RichLog { text-style: bold;  }
    """

    def compose(self) -> ComposeResult:
        #yield Header()
        with Grid(id="main_grid"):
            # Left side: Configuration
            with Vertical(id="config_panel"):
                with Collapsible(title="1. Profile (Problems)"):
                    #yield Label("1. Profile (Problems)")
                    yield ListView(id="profile_list")

                with Collapsible(title="2. Parameters (Logic)"):
                    #yield Label("2. Parameters (Logic)")
                    yield ListView(id="params_list")

                with Collapsible(title="3. Algorithms"):
                    #yield Label("3. Algorithms")
                    yield ListView(id="algos_list")

                yield Button("🚀 Run Experiment", id="launch")
            
            # Right side: Live Logs
            with Vertical(id="log_panel"):
                yield Label("📜 Experiment Output")
                yield RichLog(id="log_window", highlight=True, markup=True, wrap=True)
                
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.populate_list("profile_list", "profile_")
        self.populate_list("params_list", "params_")
        self.populate_list("algos_list", "algos")
        self.log_widget = self.query_one("#log_window", RichLog)

    def populate_list(self, list_id, prefix):
        list_view = self.query_one(f"#{list_id}", ListView)
        files = [f for f in os.listdir("configs") if f.startswith(prefix) and f.endswith(".yaml")]
        for f in sorted(files):
            list_view.append(ListItem(Label(f)))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch":
            # 1. Gather selections
            profile = self.query_one("#profile_list", ListView).highlighted_child.query_one(Label).visual
            params = self.query_one("#params_list", ListView).highlighted_child.query_one(Label).visual
            algos = self.query_one("#algos_list", ListView).highlighted_child.query_one(Label).visual

            self.log_widget.clear()
            self.log_widget.write(f"[bold green]Starting experiment with {profile}...[/bold green]\n")
            
            # 2. Run as a non-blocking process
            cmd = f"./.pymoo-venv/bin/python nibea_test.py --profile configs/{profile} --params configs/{params} --algos configs/{algos}"
            
            # Use Textual's run_process to stream stdout to the log window
            #self.run_experiment(cmd) #TODO: not working for now
            with self.suspend():
                self.log_widget.write(f"{cmd}\n")
                os.system(cmd)

    async def run_experiment(self, cmd: str) -> None:
        self.query_one("#launch").disabled = True
        
        # Start the subprocess and pipe output
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Read line by line
        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if line:
                    self.log_widget.write(line.decode().strip())
                else:
                    break

        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr))
        await process.wait()
        
        self.log_widget.write("\n[bold cyan]Experiment Complete![/bold cyan]")
        self.query_one("#launch").disabled = False

if __name__ == "__main__":
    app = ExperimentLauncher()
    app.run()
