from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("taslow_email_extraction_agent.app:app", host="127.0.0.1", port=8087, reload=True)


if __name__ == "__main__":
    main()
