"""Export corrected-label ESD panels using the archived probabilities and plot tables."""
import argparse
from pathlib import Path
import sys


def render(project, output):
    import numpy as np
    import pandas as pd
    stage23 = project / "23_manuscript_figures"
    source = stage23 / "relabel_diagnostic_update"
    original = pd.read_csv(source/"tables/ESD_scores_original.csv", float_precision="round_trip")
    corrected = pd.read_csv(source/"tables/ESD_scores_reversed.csv", float_precision="round_trip")
    keys = ["cohort", "rank"]
    original = original.sort_values(keys)
    corrected = corrected.sort_values(keys)
    if not original[keys].equals(corrected[keys]):
        raise ValueError("ESD plotting cohorts or ranks differ")
    if not np.array_equal(original.probability, corrected.probability):
        raise ValueError("Frozen probabilities differ between label versions")
    if not np.array_equal(corrected.LNM, 1-original.LNM):
        raise ValueError("The supplied corrected case labels do not match the audited complement")
    sys.path.insert(0, str(source/"scripts"))
    import draw_relabel_esd as plots

    class CorrectedFrame(plots.ScenarioFrame):
        def header(self, letter, title, note=None, note_color=plots.MUTED):
            if letter in ("C", "F") and title in ("Confusion matrix", "Ranked ESD scores"):
                title += f" ({self.cutoff:.3f})"
            plots.Frame.header(self, letter, title, "ESD validation · corrected LNM labels", plots.ORANGE)

    def panels(scenario, kind, functions, sizes, layout, height, cutoff, positive):
        if scenario != "reversed":
            raise ValueError("Expected the supplied corrected-label table")
        for letter, function in functions.items():
            width, height = sizes[letter]
            fig = plots.plt.figure(figsize=(width*plots.MM, height*plots.MM))
            function(CorrectedFrame(fig, (2, 1, width-4, height-2), scenario, cutoff), True)
            for ax in fig.axes:
                if ax.get_ylabel() == "Hypothetical LNM fraction":
                    ax.set_ylabel("Observed LNM fraction")
            plots.export(fig, output, kind+letter)

    output.mkdir(parents=True, exist_ok=True)
    plots.ScenarioFrame = CorrectedFrame
    plots.render = panels
    plots.draw_figure4("reversed")
    plots.draw_supplement("reversed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-corrected-esd-labels", action="store_true", required=True,
                        help="Data custodian confirms the archived complemented case labels")
    args = parser.parse_args()
    render(args.project_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
