import os
import threading

from services.classification.app.logic.explainerengine import ExplainerEngine


class ExplainerRunner:
    def __init__(self, engine: ExplainerEngine):
        self.engine = engine
        self._thread = None
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", 8050))

    def start_dashboard(self, host=None, port=None):
        host = host or self.host
        port = port or self.port

        if not self.engine.dashboard:
            raise Exception("Dashboard not initialized")

        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self.engine.dashboard.run, kwargs={"host": host, "port": port, "debug": False}, daemon=True
            )
            self._thread.start()
            print(f"📊 Dashboard running at http://{host}:{port}")
        else:
            print("⚠️ Dashboard already running.")
