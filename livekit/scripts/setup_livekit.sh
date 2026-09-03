#!/bin/bash
# One-time LiveKit side of the SIP wiring. Run AFTER `lk cloud auth`.
# Creates the inbound trunk (matching our Twilio number) and the dispatch
# rule that sends each call to a fresh room with the pharmacy-livekit agent.
set -euo pipefail
cd "$(dirname "$0")/.."

NUMBER=$(grep '^PHARMACY_LK_NUMBER=' .env | cut -d= -f2)
echo "number: $NUMBER"

cat > ops/inbound-trunk.json << EOF
{ "trunk": { "name": "pharmacy-livekit inbound", "numbers": ["$NUMBER"] } }
EOF

cat > ops/dispatch-rule.json << 'EOF'
{
  "dispatch_rule": {
    "name": "pharmacy-livekit dispatch",
    "rule": { "dispatchRuleIndividual": { "roomPrefix": "pharmacy-" } },
    "roomConfig": { "agents": [{ "agentName": "pharmacy-livekit" }] }
  }
}
EOF

echo "--- creating inbound trunk"
lk sip inbound create ops/inbound-trunk.json
echo "--- creating dispatch rule"
lk sip dispatch create ops/dispatch-rule.json
echo "--- done. SIP host for the Twilio side:"
lk project list --json | python3 -c "import json,sys; [print(p['ProjectId'].removeprefix('p_') + '.sip.livekit.cloud') for p in json.load(sys.stdin)]"
