"""Canonical (raw + normalized) objects from validated records; shared with benchmarks."""

import logging

from backend.app.schemas.domain import Customer, KYCRecord
from data_pipeline.consolidation.models import CanonicalCustomer, CanonicalKYCDocument
from data_pipeline.normalization.addresses import normalize_address
from data_pipeline.normalization.identifiers import normalize_document_number
from data_pipeline.normalization.models import NormalizedName
from data_pipeline.normalization.names import normalize_name

logger = logging.getLogger(__name__)


def canonicalize_customer(customer: Customer) -> CanonicalCustomer:
    """Raises ValueError if the primary name has no usable tokens (caller quarantines)."""
    alternates: list[NormalizedName] = []
    for alt in customer.alternate_names:
        try:
            alternates.append(normalize_name(alt))
        except ValueError:
            logger.warning("dropping unusable alternate name for %s", customer.customer_id)
    return CanonicalCustomer(
        customer=customer,
        name=normalize_name(customer.name),
        alternate_names=alternates,
        address=normalize_address(customer.address),
    )


def canonicalize_kyc(record: KYCRecord) -> CanonicalKYCDocument:
    """Raises ValueError if the name or document number is unusable (caller quarantines)."""
    return CanonicalKYCDocument(
        record=record,
        name=normalize_name(record.name),
        address=normalize_address(record.address),
        document_number=normalize_document_number(record.document_number),
    )
