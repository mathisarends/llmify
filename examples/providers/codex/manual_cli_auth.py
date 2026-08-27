"""Inspect the borrowed Codex CLI credentials and wire them up by hand.

`ChatCodex.from_cli()` does all of this in one line — the explicit form
is for pointing at a different `auth.json` or sharing one token provider
across several clients.
"""

import asyncio

from llmify import ChatCodex, CodexCliAuth, UserMessage
from llmify.providers.codex import codex_auth_path, codex_home, read_codex_credentials


async def main() -> None:
    print(f"Codex home: {codex_home()}")
    print(f"Auth file:  {codex_auth_path()} (exists: {codex_auth_path().exists()})")

    credentials = read_codex_credentials()
    auth = CodexCliAuth(credentials)

    print(f"Loaded from: {credentials.auth_path}")
    print(f"Account ID:  {credentials.account_id or '-'}")
    print(f"Fresh:       {credentials.is_fresh}")
    if credentials.expires_in is not None:
        print(f"Expires in:  {credentials.expires_in / 60:.1f} min")

    # Refreshes the token first if it is about to expire.
    token = await auth()
    print(f"Token:       {token[:12]}...{token[-6:]} (len {len(token)})")

    llm = ChatCodex(
        model="gpt-5.6-terra",
        api_key=auth,
        chatgpt_account_id=auth.account_id,
        temperature=0.2,
    )

    response = await llm.invoke([UserMessage(content="Say hi in one word.")])
    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
