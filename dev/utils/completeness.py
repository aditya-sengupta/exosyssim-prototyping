'''
Script to generate completeness contours.
'''

from dfm import *

bins = "hsu" # "hsu" or "dfm"

if bins == "dfm":
    period_rng = (50, 300)
    rp_rng = (0.75, 2.5)
    period = np.linspace(period_rng[0], period_rng[1], 57)
    rp = np.linspace(rp_rng[0], rp_rng[1], 61)
elif bins == "hsu":
    period = np.array([0.5, 1.25, 2.5, 5, 10, 20, 40, 80, 160, 320])
    rp = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 12, 16])
    period_rng = (min(period), max(period))
    rp_rng = (min(rp), max(rp))

period_grid, rp_grid = np.meshgrid(period, rp, indexing="ij")

stlr = get_stellar()
stlr = stlr[np.isfinite(stlr.mass)]
stlr = stellar_cuts(stlr, cut_to_gk=(bins == "dfm"))

# kois = get_kois()
# kois = kois_cuts(kois[kois["kepid"].isin(stlr["kepid"])], period_rng, rp_rng)

make_comp(stlr, period_grid, rp_grid, name=bins)
