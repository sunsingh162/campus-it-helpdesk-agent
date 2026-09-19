from ..state import TicketState


def _make_stub(label: str):
    def node(state: TicketState) -> dict:
        return {
            "response": f"[{label} stub] Placeholder response for: {state['ticket_text']!r}"
        }

    node.__name__ = f"{label.lower()}_stub"
    return node


password_stub = _make_stub("Password")
network_stub = _make_stub("Network")
hardware_stub = _make_stub("Hardware")
account_stub = _make_stub("Account")
unknown_stub = _make_stub("Unknown")
