"""Generate synthetic contracts, chunk, embed (BGE), and store in pgvector.

Blueprint §2.1.3 | Stage 2.

Deviation from blueprint: contracts are stored as .txt files instead of .pdf
(fpdf2/reportlab not in project requirements). Embedding pipeline is identical.
See Project_Notes.md ADR-005.

Run after seed_adventureworks.py. Requires BGE-large-en-v1.5 (~1.3 GB download
on first run). Embedding on CPU takes ~3-5 min for 20 contracts.

Usage:
    python -m backend.scripts.seed_contracts
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import asyncpg
import numpy as np
from app.config import get_settings
from app.rag.chunker import chunk_text as chunk_text
from app.rag.embedder import embed_batch
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[2]
_CONTRACTS_DIR = _REPO_ROOT / "data" / "contracts"

# ---------------------------------------------------------------------------
# Force Majeure variants (critical for CRAG evaluation — Blueprint §2.1.3)
# Variant is chosen by: (contract_id - 1) % 5
# ---------------------------------------------------------------------------

_FM_VARIANTS: list[str] = [
    # Variant 0 — Standard FM
    """\
SECTION 18 — FORCE MAJEURE

18.1  Neither party shall be liable for any failure or delay in performance \
to the extent that such failure or delay is caused by circumstances beyond \
that party's reasonable control, including but not limited to: acts of God, \
war, armed conflict, terrorism, civil unrest, fire, flood, earthquake, \
explosion, epidemic, labour strike or lockout, or any act or omission of \
any governmental authority ("Force Majeure Event").

18.2  The party affected shall notify the other party in writing within ten \
(10) business days of the occurrence of a Force Majeure Event, describing \
the nature of the event, its expected duration, and the steps being taken \
to mitigate its effects.

18.3  The obligations of the affected party shall be suspended for the \
duration of the Force Majeure Event. Both parties shall use commercially \
reasonable efforts to resume performance as soon as practicable.

18.4  If a Force Majeure Event continues for more than one hundred eighty \
(180) days, either party may terminate this Agreement upon thirty (30) days \
written notice without liability to the other party.""",
    # Variant 1 — Extended FM with Pandemic clause
    """\
SECTION 18 — FORCE MAJEURE

18.1  Neither party shall be liable for any failure or delay in performance \
caused by a Force Majeure Event. For the purposes of this Agreement, a \
"Force Majeure Event" includes: acts of God, war, terrorism, fire, flood, \
earthquake, natural disaster, epidemic, pandemic, public health emergency \
declared by the World Health Organization (WHO) or a national government, \
governmental sanctions or trade embargoes, quarantine restrictions, and \
actions of public authorities.

18.2  Pandemic-Specific Provisions. In the event of a pandemic or public \
health emergency, the Supplier shall: (a) activate its documented pandemic \
response plan within forty-eight (48) hours of a WHO declaration; \
(b) designate an alternate supply chain route; and (c) provide the Buyer \
with weekly status updates. Pricing may be renegotiated if raw material \
costs increase by more than fifteen percent (15%) due to pandemic-related \
supply disruptions.

18.3  The affected party shall provide written notice within five (5) \
business days of the onset of a Force Majeure Event. Failure to provide \
timely notice shall result in forfeiture of Force Majeure protections for \
delays preceding the notice.

18.4  No Force Majeure Event shall excuse payment obligations for goods \
already delivered.""",
    # Variant 2 — FM with automatic price-adjustment trigger
    """\
SECTION 18 — FORCE MAJEURE

18.1  A "Force Majeure Event" means any event beyond the reasonable control \
of a party, including acts of God, labour disputes, governmental actions, \
fire, flood, natural disaster, or war.

18.2  Automatic Price-Adjustment Right. Where a Force Majeure Event causes \
a sustained increase in the Supplier's raw material costs exceeding ten \
percent (10%) for a consecutive period of thirty (30) days or more, the \
Supplier shall have the right to initiate a price renegotiation. The Buyer \
must respond to any such request within fifteen (15) business days. Should \
the parties fail to agree on revised pricing within thirty (30) days of \
the renegotiation request, either party may terminate the affected purchase \
orders without penalty.

18.3  Notice of a Force Majeure Event must be given in writing within seven \
(7) calendar days of occurrence. Notices shall include a written estimate \
of expected duration and impact on delivery schedules.

18.4  Mitigation obligation: the affected party shall take all commercially \
reasonable steps to minimise the impact and duration of any Force Majeure \
Event, including sourcing materials from alternative suppliers.""",
    # Variant 3 — FM limited to 90 days then termination right
    """\
SECTION 18 — FORCE MAJEURE

18.1  Definition. A "Force Majeure Event" means any event or circumstance \
beyond a party's reasonable control that prevents or delays performance, \
including but not limited to: natural disasters, acts of war or terrorism, \
civil disturbance, governmental restrictions, strikes, or pandemics.

18.2  Notice Requirement. The party claiming Force Majeure relief must \
provide written notice to the other party within forty-eight (48) hours of \
the Force Majeure Event. Such notice must describe: (i) the nature of the \
event; (ii) its expected duration; and (iii) the specific obligations \
affected. Failure to provide timely notice shall result in the claiming \
party bearing liability for any resulting damages.

18.3  90-Day Limit. Force Majeure relief shall not excuse non-performance \
for a period exceeding ninety (90) calendar days. If the Force Majeure \
Event continues beyond ninety (90) days, the non-affected party shall have \
the right to terminate this Agreement upon seven (7) days written notice, \
without liability for termination fees, penalties, or damages.

18.4  Partial Performance. Where a Force Majeure Event affects only part \
of the Supplier's obligations, the Supplier shall continue to fulfil all \
unaffected obligations at the prices and on the terms specified herein.""",
    # Variant 4 — BCP (Business Continuity Plan) obligation
    """\
SECTION 18 — FORCE MAJEURE

18.1  Force Majeure. Neither party shall be in breach of this Agreement if \
performance is prevented or impeded by a Force Majeure Event. Events \
qualifying as Force Majeure include: acts of nature, war, terrorism, riots, \
fire, flood, earthquake, governmental action, power outages of more than \
seventy-two (72) hours, or epidemic/pandemic declared by competent \
authorities.

18.2  Business Continuity Obligation. The Supplier represents and warrants \
that it maintains a documented Business Continuity Plan ("BCP") addressing \
supply chain disruptions. The Supplier shall: \
(a) provide a copy of its current BCP to the Buyer upon request; \
(b) notify the Buyer within forty-eight (48) hours of activating its BCP; \
(c) provide monthly BCP status reports during any active Force Majeure \
Event; and (d) conduct a post-event review within sixty (60) days of \
resuming normal operations.

18.3  Supplier's failure to maintain or activate a BCP in accordance with \
industry standards shall constitute a material breach of this Agreement and \
shall negate any Force Majeure defence available to the Supplier.

18.4  Audit Right. The Buyer reserves the right to audit the Supplier's \
BCP compliance annually and during any active Force Majeure Event.""",
]

# ---------------------------------------------------------------------------
# Contract text template
# ---------------------------------------------------------------------------


def _generate_contract_text(
    contract_id: int,
    supplier_name: str,
    effective_date: str,
    expiry_date: str,
    account_number: str,
) -> str:
    """Generate a synthetic supply contract text (~1 500 words, structured sections)."""
    fm_variant = (contract_id - 1) % 5
    fm_text = _FM_VARIANTS[fm_variant]

    # Vary Incoterms based on the FIRST token of account_number (e.g. "TQ", "FAB")
    incoterms_map = {
        "TQ": "DAP (Delivered At Place)",  # TQ-Electronics, TQ-Sub
        "FAB": "FOB (Free On Board)",  # Fabrikam Bearings
        "LUC": "CIF (Cost, Insurance and Freight)",  # Lucerne Metals
        "CON": "EXW (Ex Works)",  # Contoso Polymers
        "AWC": "DDP (Delivered Duty Paid)",  # AWC Direct
        "HAN": "FCA (Free Carrier)",  # Hanover Steel
        "SHA": "EXW (Ex Works)",  # Shanghai Polymers
        "PAC": "CFR (Cost and Freight)",  # Pacific Rubber
        "NOR": "CIF (Cost, Insurance and Freight)",  # Nordic Precision
        "AUS": "FOB (Free On Board)",  # Australian Iron Ore
        "CAN": "CFR (Cost and Freight)",  # Canadian Aluminum
        "BRA": "FOB (Free On Board)",  # Borracha Brasileira
        "KOR": "DAP (Delivered At Place)",  # Korea Semiconductor
    }
    incoterms_key = account_number.split("-")[0] if "-" in account_number else "AWC"
    incoterms = incoterms_map.get(incoterms_key, "DAP (Delivered At Place)")

    governing_law = "the State of Washington, USA"
    if "DE" in account_number or "STL" in account_number or "SUB" in account_number:
        governing_law = "the laws of Germany"
    elif "CH" in account_number or "MET" in account_number:
        governing_law = "the laws of Switzerland"
    elif "CN" in account_number or "SHA" in account_number or "POL" in account_number:
        governing_law = "the laws of England and Wales"
    elif "SE" in account_number or "PRE" in account_number:
        governing_law = "the laws of Sweden"

    return f"""\
SUPPLY AGREEMENT
================
Contract No.: {contract_id:04d}
Date:         {effective_date}
Buyer:        Adventure Works Cycles, Inc., 1 Microsoft Way, Redmond, WA 98052, USA
Supplier:     {supplier_name}
Account:      {account_number}
Effective:    {effective_date}
Expiry:       {expiry_date}

RECITALS

WHEREAS, Buyer desires to purchase certain goods and components from Supplier;
WHEREAS, Supplier desires to supply such goods on the terms and conditions set forth herein;
NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

SECTION 1 — DEFINITIONS

1.1  "Agreement" means this Supply Agreement together with all Exhibits attached hereto.
1.2  "Goods" means all components, materials, and assemblies listed in Exhibit A — Product Schedule.
1.3  "Purchase Order" or "PO" means a written order issued by Buyer referencing this Agreement.
1.4  "Delivery Date" means the date specified in the applicable Purchase Order.
1.5  "Specifications" means the technical requirements set forth in Exhibit B — Technical Specifications.
1.6  "Confidential Information" means any non-public information disclosed by one party to the other.
1.7  "Intellectual Property" means patents, trademarks, trade secrets, copyrights, and related rights.

SECTION 2 — TERM AND RENEWAL

2.1  Initial Term. This Agreement shall commence on {effective_date} and shall continue in full force \
and effect until {expiry_date} (the "Initial Term"), unless terminated earlier in accordance with \
Section 16.
2.2  Renewal. Upon expiry of the Initial Term, this Agreement shall automatically renew for successive \
one-year periods unless either party provides written notice of non-renewal at least ninety (90) days \
prior to the end of the then-current term.
2.3  Survival. Sections 9 (Confidentiality), 14 (Limitation of Liability), and 20 (Governing Law) shall \
survive termination or expiry of this Agreement.

SECTION 3 — PRICING AND VOLUME COMMITMENTS

3.1  Pricing. The prices for Goods shall be as set forth in the Purchase Order, which shall be consistent \
with Exhibit A. Prices are in United States Dollars (USD) unless otherwise specified.
3.2  Price Stability. Supplier guarantees that prices shall remain fixed for the first twelve (12) months \
of the Initial Term. Thereafter, Supplier may propose price adjustments of no more than three percent (3%) \
per annum, provided sixty (60) days written notice is given.
3.3  Volume Discounts. Buyer shall be entitled to volume discounts as follows: (a) 2% discount on annual \
spend exceeding USD 500,000; (b) 4% discount on annual spend exceeding USD 1,000,000.
3.4  Most Favored Customer. Supplier represents that the prices offered hereunder are no less favourable \
than those offered to any similarly-situated customer for comparable quantities and quality of Goods.

SECTION 4 — PAYMENT TERMS

4.1  Standard Terms. Payment shall be due net thirty (30) days from the date of Supplier's invoice, \
provided that the invoice is accurate and complete.
4.2  Disputes. Buyer may withhold payment of disputed amounts, provided that Buyer notifies Supplier in \
writing within ten (10) business days of receipt of the invoice.
4.3  Late Payment. Undisputed amounts not paid within thirty (30) days shall bear interest at one and \
one-half percent (1.5%) per month, or the maximum rate permitted by applicable law, whichever is lower.
4.4  Currency. All invoices shall be denominated and paid in United States Dollars unless mutually agreed \
in writing.

SECTION 5 — SHIPMENT AND DELIVERY

5.1  Incoterms. Goods shall be delivered {incoterms} to Buyer's designated facility.
5.2  Packing. Supplier shall pack all Goods in a manner suitable for safe transport, using materials \
consistent with industry standards for the type of Good shipped.
5.3  Risk of Loss. Risk of loss and title to Goods shall transfer to Buyer in accordance with the \
applicable Incoterms designation.
5.4  Shipping Documentation. Each shipment shall be accompanied by a packing slip, certificate of \
conformance, and, where applicable, a certificate of origin.
5.5  Partial Shipments. Partial shipments are permissible only with Buyer's prior written consent.

SECTION 10 — QUALITY REQUIREMENTS

10.1  Standards. All Goods shall conform to: (a) the Specifications set forth in Exhibit B; \
(b) all applicable ISO standards (including ISO 9001:2015); and (c) all applicable laws and regulations.
10.2  Quality Management System. Supplier shall maintain a quality management system certified to \
ISO 9001:2015 or equivalent throughout the term of this Agreement.
10.3  Statistical Process Control. For high-volume components (annual volume > 10,000 units), Supplier \
shall implement statistical process control measures and provide monthly Cpk reports to Buyer.
10.4  Non-Conforming Goods. Supplier shall immediately notify Buyer upon discovery of any non-conformance \
in Goods already shipped, including the quantity shipped, the extent of non-conformance, and the \
corrective action taken or planned.
10.5  Traceability. Supplier shall maintain full batch/lot traceability for all Goods for a minimum of \
five (5) years from the date of shipment.

SECTION 11 — INSPECTION RIGHTS

11.1  Incoming Inspection. Buyer shall have the right to inspect all Goods within fifteen (15) business \
days of receipt.
11.2  On-Site Audits. Buyer may, upon ten (10) business days notice, audit Supplier's manufacturing \
facilities and quality records up to two (2) times per calendar year.
11.3  Source Inspection. For Goods valued in excess of USD 50,000 per Purchase Order, Buyer reserves \
the right to conduct source inspection prior to shipment.

SECTION 12 — NON-CONFORMING GOODS

12.1  Rejection. Buyer may reject Goods that do not conform to the Specifications within fifteen (15) \
business days of inspection.
12.2  Replacement. Upon receipt of notice of rejection, Supplier shall, at Buyer's election: \
(a) replace non-conforming Goods within thirty (30) days; or (b) issue a full credit note.
12.3  Costs. All costs associated with the return, replacement, or disposal of non-conforming Goods \
shall be borne by Supplier.

SECTION 14 — LIMITATION OF LIABILITY

14.1  Exclusion of Consequential Damages. IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR \
ANY INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO \
THIS AGREEMENT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
14.2  Cap on Liability. EACH PARTY'S TOTAL CUMULATIVE LIABILITY ARISING OUT OF OR RELATED TO THIS \
AGREEMENT SHALL NOT EXCEED THE TOTAL AMOUNTS PAID OR PAYABLE UNDER THIS AGREEMENT IN THE TWELVE \
(12) MONTHS PRECEDING THE CLAIM.
14.3  Exceptions. The limitations in Sections 14.1 and 14.2 shall not apply to: (a) breaches of \
confidentiality; (b) intellectual property infringement; (c) gross negligence or wilful misconduct; \
or (d) any claims for personal injury or death.

SECTION 16 — TERMINATION

16.1  Termination for Convenience. Either party may terminate this Agreement at any time upon ninety \
(90) days written notice to the other party.
16.2  Termination for Cause. Either party may terminate this Agreement immediately upon written notice \
if the other party: (a) commits a material breach and fails to cure such breach within thirty (30) days \
of written notice; (b) becomes insolvent, makes an assignment for the benefit of creditors, or is subject \
to bankruptcy proceedings; or (c) engages in fraudulent, corrupt, or illegal activity.
16.3  Effect of Termination. Upon termination: (a) all outstanding Purchase Orders shall be cancelled, \
subject to payment for Goods already shipped; (b) each party shall return or destroy the other's \
Confidential Information; and (c) all licence rights granted hereunder shall terminate immediately.

{fm_text}

SECTION 20 — GOVERNING LAW AND DISPUTE RESOLUTION

20.1  Governing Law. This Agreement shall be governed by and construed in accordance with \
{governing_law}, without regard to its conflict-of-laws principles.
20.2  Negotiation. The parties shall first attempt to resolve any dispute through good-faith negotiation \
between senior representatives within thirty (30) days of written notice of a dispute.
20.3  Arbitration. If negotiation fails, disputes shall be finally resolved by binding arbitration under \
the rules of the International Chamber of Commerce (ICC). The arbitration shall take place in Geneva, \
Switzerland, and shall be conducted in the English language.
20.4  Injunctive Relief. Notwithstanding Section 20.3, either party may seek injunctive or equitable \
relief from a court of competent jurisdiction to prevent irreparable harm.
20.5  Class Action Waiver. Each party waives any right to participate in a class action or collective \
proceeding related to this Agreement.

EXHIBIT A — PRODUCT SCHEDULE
See attached Purchase Order schedule, updated quarterly.

EXHIBIT B — TECHNICAL SPECIFICATIONS
See Supplier Quality Manual v{contract_id}.0, incorporated herein by reference.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

ADVENTURE WORKS CYCLES, INC.
By: ______________________________
Name: Chief Procurement Officer

{supplier_name.upper()}
By: ______________________________
Name: Authorised Signatory
Date: {effective_date}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Generate contract texts, chunk, embed (BGE), and insert into pgvector."""
    settings = get_settings()
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    logger.info("Connecting to PostgreSQL…")
    conn: asyncpg.Connection = await asyncpg.connect(dsn)
    await register_vector(conn)

    try:
        # Fetch contract records seeded by seed_adventureworks.py
        rows = await conn.fetch("""
            SELECT c.contract_id, c.contract_pdf_path, c.effective_date, c.expiry_date,
                   v.name AS supplier_name, v.account_number
            FROM supply_chain.contracts c
            JOIN purchasing.vendor v ON v.business_entity_id = c.supplier_id
            ORDER BY c.contract_id
            """)
        if not rows:
            raise RuntimeError(
                "No contracts found in supply_chain.contracts. "
                "Run seed_adventureworks.py first."
            )

        logger.info("Found %d contracts to process.", len(rows))
        _CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

        all_chunks: list[tuple[int, int, str]] = []  # (contract_id, chunk_index, text)

        # Step 1: generate text files and collect chunks
        for row in rows:
            cid: int = row["contract_id"]
            txt_path = Path(row["contract_pdf_path"])
            eff = str(row["effective_date"])
            exp = str(row["expiry_date"])

            # Check if already embedded
            existing = await conn.fetchval(
                "SELECT COUNT(*) FROM supply_chain.contract_embeddings WHERE contract_id = $1",
                cid,
            )
            if existing > 0:
                logger.info(
                    "Contract %d already embedded (%d chunks) — skipping.",
                    cid,
                    existing,
                )
                continue

            # Generate contract text
            text = _generate_contract_text(
                contract_id=cid,
                supplier_name=row["supplier_name"],
                effective_date=eff,
                expiry_date=exp,
                account_number=row["account_number"],
            )

            # Write text file
            txt_path.write_text(text, encoding="utf-8")
            logger.debug("Wrote %s (%d chars)", txt_path.name, len(text))

            # Chunk
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append((cid, idx, chunk))

        if not all_chunks:
            logger.info("All contracts already embedded. Nothing to do.")
            return

        # Step 2: batch embed  (BGE downloads ~1.3 GB on first run)
        logger.info(
            "Embedding %d chunks via BGE-large-en-v1.5 (CPU — this may take a few minutes)…",
            len(all_chunks),
        )
        texts = [c[2] for c in all_chunks]
        vectors = embed_batch(texts)  # list[list[float]], each 1024-dim
        logger.info("Embedding complete.")

        # Step 3: insert into supply_chain.contract_embeddings
        logger.info("Inserting %d chunk embeddings into pgvector…", len(all_chunks))
        first_chunk_ids: dict[int, uuid.UUID] = {}  # contract_id → first chunk UUID

        for (cid, chunk_idx, chunk_text_), vector in zip(all_chunks, vectors):
            chunk_uuid = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO supply_chain.contract_embeddings
                    (id, contract_id, chunk_index, chunk_text, embedding)
                VALUES ($1, $2, $3, $4, $5)
                """,
                chunk_uuid,
                cid,
                chunk_idx,
                chunk_text_,
                np.array(vector, dtype=np.float32),
            )
            if chunk_idx == 0:
                first_chunk_ids[cid] = chunk_uuid

        # Step 4: update contract.embedding_id with first chunk UUID
        for cid, first_uuid in first_chunk_ids.items():
            await conn.execute(
                "UPDATE supply_chain.contracts SET embedding_id = $1 WHERE contract_id = $2",
                first_uuid,
                cid,
            )

        logger.info(
            "✅ Contracts seeded — %d contracts, %d chunks embedded.",
            len(first_chunk_ids),
            len(all_chunks),
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )
    asyncio.run(main())
