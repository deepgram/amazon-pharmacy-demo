"""One-time Twilio side of the SIP wiring: create an Elastic SIP trunk, point
its origination at LiveKit Cloud, and move the demo number onto it.

  uv run python scripts/setup_twilio_trunk.py <livekit-sip-host>
  e.g. uv run python scripts/setup_twilio_trunk.py abc123xyz.sip.livekit.cloud

Get the host from `lk project list --json`: strip the `p_` prefix from
ProjectId -> <id>.sip.livekit.cloud. Idempotent: reuses an existing trunk
named 'pharmacy-livekit'. Note: associating a number with a trunk clears its
voice webhook config (fine — this number is dedicated to this demo).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from twilio.rest import Client

TRUNK_NAME = "pharmacy-livekit"
DOMAIN = "dg-pharmacy-livekit.pstn.twilio.com"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sip_host = sys.argv[1].replace("sip:", "").split(";")[0]
    number = os.environ["PHARMACY_LK_NUMBER"]
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    trunk = next((t for t in client.trunking.v1.trunks.list() if t.friendly_name == TRUNK_NAME), None)
    if trunk is None:
        trunk = client.trunking.v1.trunks.create(friendly_name=TRUNK_NAME, domain_name=DOMAIN)
        print(f"trunk created: {trunk.sid}")
    else:
        print(f"trunk exists: {trunk.sid}")

    sip_url = f"sip:{sip_host};transport=tcp"
    existing = client.trunking.v1.trunks(trunk.sid).origination_urls.list()
    if not any(o.sip_url == sip_url for o in existing):
        for o in existing:  # replace stale targets
            client.trunking.v1.trunks(trunk.sid).origination_urls(o.sid).delete()
        client.trunking.v1.trunks(trunk.sid).origination_urls.create(
            friendly_name="LiveKit Cloud SIP", sip_url=sip_url, weight=1, priority=1, enabled=True
        )
        print(f"origination set: {sip_url}")
    else:
        print(f"origination already set: {sip_url}")

    nums = client.incoming_phone_numbers.list(phone_number=number, limit=1)
    if not nums:
        sys.exit(f"{number} not found on this Twilio account")
    already = [p.sid for p in client.trunking.v1.trunks(trunk.sid).phone_numbers.list()]
    if nums[0].sid not in already:
        client.trunking.v1.trunks(trunk.sid).phone_numbers.create(phone_number_sid=nums[0].sid)
        print(f"number attached: {number}")
    else:
        print(f"number already attached: {number}")
    print("Twilio side done.")


if __name__ == "__main__":
    main()
