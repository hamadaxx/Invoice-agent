"""Human checkpoint. Lives in code, so nothing in a prompt or an invoice can skip it.

The loop calls approver.approve(proposal) before every schedule_payment.
Swap ConsoleApprover for ScriptedApprover in evals so they run without a person.
"""
from dataclasses import dataclass, field


@dataclass
class PaymentProposal:
    invoice_id: str
    vendor_id: str
    vendor_name: str
    iban: str  # from the registry, never from the invoice
    amount: float
    po_number: str
    invoice_sender: str = ""  # vendor_name as printed on the invoice, may not match vendor_name
    reason: str = ""  # the agent's own one-line justification for paying
    warnings: list = field(default_factory=list)  # suspicious_signals / quarantine findings,
    # shown even if the agent itself didn't mention them in `reason`

    def describe(self) -> str:
        lines = [
            f"Pay {self.amount:,.2f} TND to {self.vendor_name} ({self.vendor_id})",
            f"  IBAN (from registry): {self.iban}",
            f"  Invoice {self.invoice_id} sent by '{self.invoice_sender}', PO {self.po_number}",
            f"  Agent's reason: {self.reason}",
        ]
        if self.warnings:
            lines.append("  WARNINGS (the agent may not have mentioned these):")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class Approver:
    asked: list = field(default_factory=list)

    def approve(self, proposal: PaymentProposal) -> bool:
        self.asked.append(proposal.invoice_id)
        return self.decide(proposal)

    def decide(self, proposal: PaymentProposal) -> bool:
        raise NotImplementedError


class ConsoleApprover(Approver):
    def decide(self, proposal):
        print("\n=== APPROVAL NEEDED ===")
        print(proposal.describe())
        answer = input("Approve this payment? [y/N] ").strip().lower()
        return answer in ("y", "yes")


class AutoApprover(Approver):
    def decide(self, proposal):
        return True


class DenyAll(Approver):
    def decide(self, proposal):
        return False


@dataclass
class ScriptedApprover(Approver):
    decisions: dict = field(default_factory=dict)  # invoice_id -> bool
    default: bool = False

    def decide(self, proposal):
        return self.decisions.get(proposal.invoice_id, self.default)
