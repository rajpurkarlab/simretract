"""Policy serving: wire protocol, plane fit and the checkpoint server.

The server (``simretract.serve.policy_server``) runs in the policy
environment and answers observation queries from the simulation evaluator
and the real-robot client, which run in their own environments.  Everything
they share lives here: the length-prefixed pickle framing
(``simretract.serve.protocol``) and the served plane fit
(``simretract.serve.plane``).
"""
