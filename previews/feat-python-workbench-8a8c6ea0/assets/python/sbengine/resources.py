"""Resource proxies: DSP demand and a unitless cost. Same structure as the 2026-09-24 explorer.

Not changed on purpose: nothing in our captured data constrains these coefficients, so they remain placeholders
(docs/ASSUMPTIONS.md). One extension: the capability count now includes one entry per supported weight-format
class (matrix.rates > 0) instead of the explorer's fixed four "types", because our graphs use mixed formats and
supporting more formats is a design choice with a cost. Two further terms are new: a per-bank L1 cost and a per-PE
weight-staging cost for the direct weight path, because the explorer priced SRAM by capacity alone and so treated L1
bandwidth as free (placeholders, docs/ASSUMPTIONS.md).
"""


def resources(cfg):
    m, v, r, d, s = cfg["matrix"], cfg["vector"], cfg["recurrent"], cfg["dma"], cfg["scalar"]
    res, cost = cfg["resources"], cfg["resources"]["cost"]
    pe = m["count"] * m["rows"] * m["cols"]
    lanes_v, lanes_r = v["count"] * v["lanes"], r["count"] * r["lanes"]
    dsp = pe * res["dsp_per_pe"] + lanes_v * res["dsp_per_vector_lane"] + lanes_r * res["dsp_per_recurrent_lane"]
    formats_supported = sum(1 for rate in m["rates"].values() if rate > 0)
    capabilities = (len(m["caps"]) + formats_supported if m["count"] else 0) + (len(v["caps"]) if v["count"] else 0)
    scratch_mib = r["count"] * r["scratch_kib"] / 1024
    total = (pe * cost["matrix_pe"] + lanes_v * cost["vector_lane"] + lanes_r * cost["recurrent_lane"]
             + (cfg["l1"]["mib"] + scratch_mib) * cost["sram_mib"] + capabilities * cost["capability"]
             + s["count"] * cost["scalar_core"] + d["count"] * cost["dma_engine"]
             + cfg["l1"]["banks"] * cost["l1_bank"]
             + (pe * cost["direct_weight_buffer_per_pe"] if cfg["memory"]["weight_path"] == "direct" else 0.0))
    limit = res["dsp_budget"] * (1 - res["shell_reserve"])
    return {"pe": pe, "vector_lanes": lanes_v, "recurrent_lanes": lanes_r, "dsp": dsp, "dsp_limit": limit,
            "cost": total, "capabilities": capabilities}
