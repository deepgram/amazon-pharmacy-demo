"""Point the Twilio number's inbound-voice webhook at this server, or put
it back.

  uv run python scripts/set_webhook.py show
  uv run python scripts/set_webhook.py set          # -> {PUBLIC_URL}/twiml-inbound
  uv run python scripts/set_webhook.py set --url X  # -> X
  uv run python scripts/set_webhook.py revert       # -> whatever 'set' saved

'set' saves the previous URL to ops/webhook_previous.txt first, so a number
shared with something else can be handed back with 'revert'.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twilio.rest import Client

from server.config import PUBLIC_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER

SAVE = Path(__file__).resolve().parent.parent / "ops" / "webhook_previous.txt"
SAVE.parent.mkdir(parents=True, exist_ok=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    nums = client.incoming_phone_numbers.list(phone_number=TWILIO_FROM_NUMBER, limit=1)
    if not nums:
        sys.exit(f"number {TWILIO_FROM_NUMBER} not found on this account")
    num = nums[0]
    print(f"number:  {num.phone_number}  ({num.sid})")
    print(f"current: {num.voice_url}")

    if cmd == "show":
        return

    if cmd == "set":
        target = None
        if "--url" in sys.argv:
            target = sys.argv[sys.argv.index("--url") + 1]
        else:
            if not PUBLIC_URL:
                sys.exit("PUBLIC_URL not set in .env")
            target = f"{PUBLIC_URL}/twiml-inbound"
        SAVE.write_text(num.voice_url or "")
        client.incoming_phone_numbers(num.sid).update(voice_url=target, voice_method="POST")
        print(f"set to:  {target}")
        print(f"(previous saved to {SAVE})")

    elif cmd == "revert":
        if not SAVE.exists() or not SAVE.read_text().strip():
            sys.exit("no saved previous URL to revert to")
        prev = SAVE.read_text().strip()
        client.incoming_phone_numbers(num.sid).update(voice_url=prev, voice_method="POST")
        print(f"reverted to: {prev}")


if __name__ == "__main__":
    main()
