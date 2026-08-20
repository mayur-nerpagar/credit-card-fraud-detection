import threading
import time
import webbrowser

import uvicorn


HOST = "127.0.0.1"
PORT = 8000


def open_browser():
    """
    Wait for FastAPI to start and then
    automatically open the application
    in the default browser.
    """

    time.sleep(2)

    webbrowser.open(
        f"http://{HOST}:{PORT}"
    )


if __name__ == "__main__":

    # Start browser-opening thread
    browser_thread = threading.Thread(
        target=open_browser,
        daemon=True
    )

    browser_thread.start()

    # Start FastAPI
    uvicorn.run(
        "src.app:app",
        host=HOST,
        port=PORT,
        reload=True
    )