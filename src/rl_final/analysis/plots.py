"""
rliable figures for the study: sample-efficiency curves, aggregate intervals,
probability of improvement, per-env panels and a performance profile.

Run:
    python -m rl_final.analysis.plots \
        --run-dir multirun/<date>/<time>
"""

import argparse
from collections import defaultdict
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from rliable import library as rly
from rliable import metrics, plot_utils

from rl_final.analysis.run_loader import load

DPI = 300  # print quality; 150 is screen-only


def legend_below(ax, ncol_max=3, fontsize=9):
    """Put the legend under the axes instead of inside them.

    rliable draws into the top-left corner by default, which is exactly where a
    sample-efficiency curve ends up. bbox_inches="tight" at save time keeps the
    out-of-axes legend in the file.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = min(ncol_max, max(1, (len(handles) + 1) // 2))
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
    )


METRICS = {
    "iqm": metrics.aggregate_iqm,
    "mean": metrics.aggregate_mean,
    "median": metrics.aggregate_median,
    "og": metrics.aggregate_optimality_gap,
}


def step_axis(ax, max_ticks=5):
    """Env-step ticks as 50k/100k rather than 50000100000 running together."""
    ax.xaxis.set_major_locator(MaxNLocator(max_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))


parser = argparse.ArgumentParser(description="rliable figures from one or more sweeps")
parser.add_argument(
    "--run-dir",
    type=Path,
    nargs="+",
    required=True,
    help="One or more Hydra output dirs, e.g. multirun/2026-08-07/23-49-39. "
    "Several are merged, so a baseline swept on one day can be compared against "
    "another condition swept later. They must not repeat an (env, condition, seed).",
)
parser.add_argument("--env", nargs="+", default=None)
parser.add_argument(
    "--success-threshold",
    type=float,
    default=0.5,
    help="eval return counted as 'solved' in the performance profile. MiniGrid pays "
    "1 - 0.9*steps/max_steps on success and 0 otherwise, and eval reports the mean over "
    "episodes, so 0.5 ~= 'solves more often than not' and 0.9 ~= 'solves nearly every one'.",
)
parser.add_argument("--metric", choices=list(METRICS), default="iqm")
parser.add_argument(
    "--reps",
    type=int,
    default=50_000,
    help="stratified-bootstrap resamples (lower it to iterate faster)",
)
parser.add_argument(
    "--baseline",
    default=None,
    help='condition every other is compared against for P(X > Y); defaults to "PPO". '
    'Use "PPO+RND" to ask whether the LLM signal adds anything beyond novelty.',
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("figures"),
    help="directory the figures are written to (one PNG per plot)",
)
args = parser.parse_args()


def simplify_labels(names):
    """Drop a coefficient from every label if doing so leaves them all distinct.

    Phase 2 pins both betas, so printing them six times says nothing -- the arms are
    already named. A beta sweep is the opposite: the coefficient IS the condition, and
    dropping it would collide two curves into one name, so it stays.
    """

    def strip(label, term):
        if " (" not in label:
            return label
        # rsplit, not split: the coefficients are always the LAST bracket, and a base
        # name can carry one of its own ("PPO+RND+LLM (warm)").
        base, rest = label.rsplit(" (", 1)
        kept = [p for p in rest.rstrip(")").split(", ") if not p.startswith(term)]
        return f"{base} ({', '.join(kept)})" if kept else base

    out = list(names)
    for term in ("\u03b2_rnd", "\u03b2_llm"):
        candidate = [strip(n, term) for n in out]
        if len(set(candidate)) == len(set(out)):
            out = candidate
    return dict(zip(names, out, strict=True))


def build_score_tensors(runs, envs=None):
    """-> (eval_steps, {condition: (num_runs, num_tasks, num_steps)}).

    Every condition and task must share ONE eval-step grid, so the intersection is
    taken across the whole set of runs, not per condition. Seeds are sorted so
    a run's position on axis 0 is stable across tasks -- rliable's stratified
    bootstrap resamples that axis, and inconsistent ordering would silently mix
    seeds between tasks.
    """
    if envs:
        runs = [r for r in runs if r.env in envs]
    if not runs:
        raise SystemExit("no runs matched -- train something first")

    eval_steps = reduce(np.intersect1d, [r.steps for r in runs])
    if eval_steps.size == 0:
        raise SystemExit("runs share no common eval steps")

    grouped = defaultdict(lambda: defaultdict(dict))  # condition -> task -> seed -> returns
    for run in runs:
        aligned = run.returns[np.isin(run.steps, eval_steps)]
        grouped[run.condition][run.env][run.seed] = aligned

    tasks = sorted({r.env for r in runs})
    score_dict = {}
    for condition, by_task in grouped.items():
        seeds = sorted(set.intersection(*(set(by_task[t]) for t in tasks if t in by_task)))
        if not seeds or len(by_task) != len(tasks):
            print(f"  ! {condition}: not present for every task/seed, skipped")
            continue
        # axis order: (run, task, step)
        score_dict[condition] = np.array(
            [[by_task[task][seed] for task in tasks] for seed in seeds]
        )
    short = simplify_labels(list(score_dict))
    score_dict = {short[k]: v for k, v in score_dict.items()}
    return eval_steps, score_dict, tasks


FINAL_WINDOW = 10  # eval points averaged into a run's "final" score


def final_scores(score_dict):
    """A run's final score: the mean of its last FINAL_WINDOW evaluations.

    Matches `hpo_objective(mode="final")` in the training code, so the figures and the
    objective measure the same thing. Taking the single last eval instead is 20 episodes
    rather than 200, and the difference is not cosmetic -- on the beta_llm sweep it
    reversed the ranking of two settings that are otherwise tied.
    """
    return {c: v[..., -FINAL_WINDOW:].mean(axis=-1) for c, v in score_dict.items()}


def sample_efficiency_curve(score_dict, eval_steps, metric="iqm", reps: int = 50_000, out=None):
    """IQM (or mean/median) per eval point, with stratified-bootstrap CIs."""
    aggregate = METRICS[metric]

    def per_step(scores):
        return np.array([aggregate(scores[..., t]) for t in range(scores.shape[-1])])

    point, cis = rly.get_interval_estimates(score_dict, per_step, reps=reps)

    plot_utils.plot_sample_efficiency_curve(
        eval_steps,
        point,
        cis,
        algorithms=list(score_dict),
        xlabel="Environment steps",
        ylabel=f"{metric.upper()} eval return",
    )
    fig = plt.gcf()
    fig.set_size_inches(7, 4.5)
    step_axis(plt.gca())
    legend_below(plt.gca())

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"wrote {out}")
    return point, cis


def aggregate_interval_plot(score_dict, reps: int = 50_000, out=None):
    """Median / IQM / Mean / Optimality Gap at the FINAL eval point, with CIs."""
    # (runs, tasks, steps) -> (runs, tasks): rliable's aggregates want a matrix.
    final = final_scores(score_dict)

    def four_metrics(scores):
        return np.array([METRICS[m](scores) for m in ("median", "iqm", "mean", "og")])

    point, cis = rly.get_interval_estimates(final, four_metrics, reps=reps)

    fig, _ = plot_utils.plot_interval_estimates(
        point,
        cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=list(final),
        xlabel="Eval return",
        xlabel_y_coordinate=-0.16,
        row_height=max(0.37, 1.6 / len(final)),
        max_ticks=3,  # 4 panels side by side: more than 3 ticks collide
    )
    # No tight_layout(): rliable does its own subplots_adjust(left=0) and puts the
    # xlabel outside the axes with fig.text, both of which tight_layout undoes.
    # bbox_inches="tight" is what keeps that out-of-axes label in the PNG.

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"wrote {out}")
    return point, cis


def probability_of_improvement_plot(score_dict, baseline=None, reps: int = 50_000, out=None):
    """P(X > Y) per condition against one baseline, with stratified-bootstrap CIs.

    The aggregate plot says each condition's score; this says how often a run of X
    beats a run of Y on the same task, which is the comparison the ablation is
    actually about. Pick `baseline="PPO+RND"` to ask whether the LLM signal adds
    anything BEYOND novelty -- against plain PPO the two bonuses are confounded.

    rliable has no get_probability_of_improvement here: PoI goes through
    get_interval_estimates with {"X,Y": (scores_x, scores_y)} pairs instead.
    """
    final = final_scores(score_dict)
    if len(final) < 2:
        print("  ! probability of improvement needs >=2 conditions, skipped")
        return None, None

    if baseline is None:  # plain PPO if it is there, else whatever sorts first
        baseline = "PPO" if "PPO" in final else sorted(final)[0]
    elif baseline not in final:
        # Labels carry their coefficients ("PPO+RND (β_rnd=0.1)"), so accept the
        # bare name too rather than making the caller type the betas exactly.
        hits = [c for c in final if c.startswith(f"{baseline} (")]
        if len(hits) != 1:
            raise SystemExit(
                f"baseline {baseline!r} matches {len(hits)} conditions; pick one of {sorted(final)}"
            )
        baseline = hits[0]

    # NOT "," -- condition_label writes "PPO+RND+LLM (β_rnd=0.1, β_llm=0.5)", and
    # rliable splits each pair key on the separator expecting exactly two halves.
    sep = " vs "
    pairs = {
        f"{cond}{sep}{baseline}": (final[cond], final[baseline])
        for cond in final
        if cond != baseline
    }
    point, cis = rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=reps)

    plot_utils.plot_probability_of_improvement(
        point,
        cis,
        pair_separator=sep,
        figsize=(5, 0.8 + 0.5 * len(pairs)),
        xlabel=f"P(X > {baseline})",
    )
    fig = plt.gcf()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"wrote {out}")
    return point, cis


def per_env_curves(score_dict, eval_steps, tasks, metric="iqm", reps: int = 50_000, out=None):
    """One learning-curve panel per environment, conditions overlaid.

    The aggregate curve pools every task into one number, which only means
    something if the tasks are comparably hard. These panels are what show
    whether a win is broad or carried by a single easy env.
    """
    aggregate = METRICS[metric]

    def per_step(scores):
        return np.array([aggregate(scores[..., t]) for t in range(scores.shape[-1])])

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.6), squeeze=False)
    for i, task in enumerate(tasks):
        # Keep the task axis (size 1) so the aggregates still get a 2-D matrix.
        one_task = {cond: scores[:, i : i + 1, :] for cond, scores in score_dict.items()}
        point, cis = rly.get_interval_estimates(one_task, per_step, reps=reps)
        plot_utils.plot_sample_efficiency_curve(
            eval_steps,
            point,
            cis,
            algorithms=list(one_task),
            ax=axes[0][i],
            xlabel="Environment steps",
            ylabel=f"{metric.upper()} eval return" if i == 0 else "",
            labelsize="medium",
            ticklabelsize="small",
        )
        step_axis(axes[0][i])
        if len(tasks) > 1:  # a single panel needs no title -- the caption says the env
            axes[0][i].set_title(task, fontsize="large")
    legend_below(axes[0][0], fontsize=8)
    fig.tight_layout()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"wrote {out}")
    return fig


def performance_profile_plot(score_dict, reps: int = 50_000, threshold=0.5, out=None):
    """Fraction of runs scoring above tau, swept across tau.

    The dashed line marks `threshold`. `evaluate()` returns the mean over eval episodes
    and a solved episode pays a bit under 1, so tau ~= the fraction solved:
    0.5 means "solves more often than not", 0.9 "solves nearly every episode".
    """
    final = final_scores(score_dict)
    taus = np.linspace(0.0, 1.0, 101)
    profiles, profile_cis = rly.create_performance_profile(final, taus, reps=reps)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_utils.plot_performance_profiles(
        profiles,
        taus,
        performance_profile_cis=profile_cis,
        xlabel=r"Eval return ($\tau$)",
        ax=ax,
    )

    i = int(np.abs(taus - threshold).argmin())
    ax.axvline(threshold, color="0.35", linestyle="--", linewidth=1, zorder=0)
    legend_below(ax)  # the per-condition fractions are printed below, not crammed into a title
    fig.tight_layout()

    print(f"\nfraction of runs with final eval return > {threshold:g}:")
    for cond in sorted(profiles):
        lo, hi = profile_cis[cond][0][i], profile_cis[cond][1][i]
        print(f"  {cond:<28} {profiles[cond][i]:.0%}   95% CI [{lo:.0%}, {hi:.0%}]")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"wrote {out}")
    return profiles, profile_cis


def main():

    runs = load(args.run_dir)
    eval_steps, score_dict, tasks = build_score_tensors(runs, args.env)

    print(f"\ntasks (num_tasks={len(tasks)}): {', '.join(tasks)}")
    print(f"eval points: {len(eval_steps)}, up to {eval_steps[-1]:,} env steps")
    for condition, scores in score_dict.items():
        print(f"  {condition:<28} tensor {scores.shape}  (runs, tasks, steps)")

    sample_efficiency_curve(
        score_dict,
        eval_steps,
        metric=args.metric,
        reps=args.reps,
        out=args.out_dir / f"sample_efficiency_{args.metric}.png",
    )
    aggregate_interval_plot(
        score_dict,
        reps=args.reps,
        out=args.out_dir / "aggregates.png",
    )
    probability_of_improvement_plot(
        score_dict,
        baseline=args.baseline,
        reps=args.reps,
        out=args.out_dir / "probability_of_improvement.png",
    )
    per_env_curves(
        score_dict,
        eval_steps,
        tasks,
        metric=args.metric,
        reps=args.reps,
        out=args.out_dir / f"per_env_{args.metric}.png",
    )
    performance_profile_plot(
        score_dict,
        reps=args.reps,
        threshold=args.success_threshold,
        out=args.out_dir / "performance_profile.png",
    )


if __name__ == "__main__":
    main()
