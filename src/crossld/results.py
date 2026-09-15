"""Result formatting: the PLINK2 tabular ``.vcor``-style output writer.


* ``r^2`` only -- ``#CHROM_A POS_A ID_A CHROM_B POS_B ID_B UNPHASED_R2``
* signed ``r`` -- the coding allele is reported for each side, so the sign is
  interpretable: ``... ID_A <MAJ|REF>_A ... ID_B <MAJ|REF>_B UNPHASED_R``
"""

from __future__ import annotations

from typing import IO

HEADER_R2 = "#CHROM_A\tPOS_A\tID_A\tCHROM_B\tPOS_B\tID_B\tUNPHASED_R2"
HEADER_R_MAJOR = (
    "#CHROM_A\tPOS_A\tID_A\tMAJ_A\tCHROM_B\tPOS_B\tID_B\tMAJ_B\tUNPHASED_R"
)
HEADER_R_REF = (
    "#CHROM_A\tPOS_A\tID_A\tREF_A\tCHROM_B\tPOS_B\tID_B\tREF_B\tUNPHASED_R"
)

# PLINK2 output prints six significant digits ("1.0" -> "1", "0.35567").
VALUE_FORMAT = "%.6g"


def format_value(value: float) -> str:
    return VALUE_FORMAT % value


def header_for(with_r: bool, allele_label: str = "MAJ") -> str:
    """Header line for the selected output shape.

    ``allele_label`` is ``"MAJ"`` for major-allele orientation and ``"REF"`` for
    ``--ref-based``; it is ignored when ``with_r`` is false.
    """

    if not with_r:
        return HEADER_R2
    if allele_label == "REF":
        return HEADER_R_REF
    if allele_label == "MAJ":
        return HEADER_R_MAJOR
    raise ValueError(f"unknown allele label: {allele_label!r}")


def format_row(
    *,
    chrom_a: str,
    pos_a: int,
    id_a: str,
    chrom_b: str,
    pos_b: int,
    id_b: str,
    r: float,
    r2: float,
    with_r: bool,
    allele_a: str = "",
    allele_b: str = "",
) -> str:
    """Render one result row without the trailing newline."""

    fields_a = [str(chrom_a), str(pos_a), str(id_a)]
    fields_b = [str(chrom_b), str(pos_b), str(id_b)]
    if with_r:
        fields_a.append(str(allele_a))
        fields_b.append(str(allele_b))
        fields = fields_a + fields_b + [format_value(r)]
    else:
        fields = fields_a + fields_b + [format_value(r2)]
    return "\t".join(fields)


def write_header(handle: IO[str], with_r: bool, allele_label: str = "MAJ") -> None:
    handle.write(header_for(with_r, allele_label) + "\n")


def write_row(
    handle: IO[str],
    *,
    with_r: bool,
    chrom_a: str,
    pos_a: int,
    id_a: str,
    chrom_b: str,
    pos_b: int,
    id_b: str,
    r: float,
    r2: float,
    allele_a: str = "",
    allele_b: str = "",
) -> None:
    handle.write(
        format_row(
            chrom_a=chrom_a,
            pos_a=pos_a,
            id_a=id_a,
            chrom_b=chrom_b,
            pos_b=pos_b,
            id_b=id_b,
            r=r,
            r2=r2,
            with_r=with_r,
            allele_a=allele_a,
            allele_b=allele_b,
        )
        + "\n"
    )


__all__ = [
    "HEADER_R2",
    "HEADER_R_MAJOR",
    "HEADER_R_REF",
    "VALUE_FORMAT",
    "format_row",
    "format_value",
    "header_for",
    "write_header",
    "write_row",
]
