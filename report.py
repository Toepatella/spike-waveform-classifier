"""Task 2: the share of narrow spikes in each analysed area, for every reporting version."""

import pandas as pd

from features import NARROW_TTP_MS
from sessions import get_session, session_from_command_line

LIKELY_CONTAMINATED = 0.20  # rp_contamination_lower above this: the 95% interval lies above 20% stray spikes


def versions(table):
    """The units in each reporting version, as name -> True/False per row: the main analysis, its
    strict and SNR-only checks, and the main analysis without the units each sensitivity marks."""
    main = table.main_analysis
    return {
        "main": main,
        "strict check": table.main_analysis_strict,
        "SNR-only check": table.main_analysis_snr_only,
        "main without likely contaminated": main & ~(table.rp_contamination_lower > LIKELY_CONTAMINATED),
        "main without near white matter": main & ~table.near_wm_limit,
        "main without Bombcell non-somatic": main & ~table.bombcell_non_somatic,
    }


def narrow_shares(table, probes):
    """One row per version, probe and analysed area: its units, how many have no trough-to-peak,
    how many are narrow, and the narrow share among the units with a trough-to-peak."""
    rows = []
    for version, members in versions(table).items():
        for probe, areas in probes.items():
            for area in areas:
                chosen = table[members & (table.probe == probe) & (table.area == area)]
                measured = int(chosen.trough_to_peak_ms.notna().sum())
                narrow = int((chosen.trough_to_peak_ms < NARROW_TTP_MS).sum())
                if measured > 0:
                    share = round(narrow / measured, 3)
                else:
                    share = float("nan")
                rows.append({
                    "version": version,
                    "probe": probe,
                    "area": area,
                    "units": len(chosen),
                    "no_ttp": len(chosen) - measured,
                    "narrow": narrow,
                    "narrow_share": share,
                })
    return pd.DataFrame(rows)


def main(session):
    units = pd.read_csv(f"outputs/good_cells_{session}.csv")
    features = pd.read_csv(f"outputs/features_{session}.csv")
    feature_columns = features[["session", "probe", "unit_id", "trough_to_peak_ms", "bombcell_non_somatic"]]
    table = units.merge(feature_columns, on=["session", "probe", "unit_id"])
    shares = narrow_shares(table, get_session(session)["probes"])
    shares.to_csv(f"outputs/narrow_shares_{session}.csv", index=False)
    print(f"Narrow: trough-to-peak below {NARROW_TTP_MS} ms")
    print(shares.to_string(index=False))


if __name__ == "__main__":
    main(session_from_command_line())
