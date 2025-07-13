import uvicorn

if __name__ == "__main__":
    # This is a programmatic way to run the Uvicorn server.
    # It gives us more control over the configuration and avoids
    # the command-line parsing issues you're seeing with the reloader.
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # Be very explicit about which folders to ignore.
        # This is more reliable than the command-line flag.
        reload_excludes=["venv", ".venv", "__pycache__"]
    )
    