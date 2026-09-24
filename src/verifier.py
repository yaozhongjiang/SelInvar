"""Verifier abstraction. Real experiments can connect VC/ZKP/oracle adapters here."""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VerificationResult:
    verified: bool
    evidence_type: str
    reason: str

class Verifier:
    def verify(self, state: Dict[str,Any], claim: Dict[str,Any]) -> VerificationResult:
        # Conservative default: claims are not verified unless a benchmark task marks them.
        if bool(claim.get('verified_credential',False)):
            return VerificationResult(True,'VC','benchmark-provided verifiable credential')
        return VerificationResult(False,'none','no machine-verifiable evidence supplied')
