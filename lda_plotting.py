import numpy as np
import matplotlib.pyplot as plt


def scatter_classes(ax, X0, X1, alpha=0.5):
    ax.scatter(X0[:, 0], X0[:, 1], alpha=alpha, label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1], alpha=alpha, label="class 1")
    ax.set(xlabel="$x_1$", ylabel="$x_2$")


def show_simulated_data(X0, X1, mu0, mu1):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X0, X1, alpha=0.6)
    ax.scatter([mu0[0], mu1[0]], [mu0[1], mu1[1]],
               marker="*", s=200, color="black", label="true means")
    ax.plot([mu0[0], mu1[0]], [mu0[1], mu1[1]], "k--", label="line between means")
    ax.set_title("Two simulated point clouds")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.show()


def show_within_class_scatter(X0, X1, m0, m1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_classes(axes[0], X0, X1, alpha=0.25)
    for X, mean, color in [(X0, m0, "#1f77b4"), (X1, m1, "#ff7f0e")]:
        axes[0].scatter(*mean, marker="*", s=200, color=color, edgecolor="black")
        for point in X[::25]:
            axes[0].annotate("", xy=point, xytext=mean,
                             arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.5})
    axes[0].set(xlim=(-4, 4), ylim=(-3, 4),
                title="Deviations from each class mean")
    axes[0].set_aspect("equal", adjustable="box")

    scatter_classes(axes[1], X0 - m0, X1 - m1, alpha=0.55)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].axvline(0, color="gray", lw=0.7)
    axes[1].scatter(0, 0, marker="*", s=200, color="black")
    axes[1].set(xlim=(-4, 4), ylim=(-4, 4),
                title="Centered clouds: variance and tilt remain")
    axes[1].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show()


def show_projection(X0, X1, c, v, point, projection, P0, P1):
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_classes(ax, X0, X1, alpha=0.15)
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    line = c + np.linspace(-4, 4, 2)[:, None] * v
    ax.plot(line[:, 0], line[:, 1], "k-", lw=1.5)
    ax.annotate("", xy=point, xytext=c,
                arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 2.5})
    ax.annotate("", xy=projection, xytext=c,
                arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 3})
    ax.plot([point[0], projection[0]], [point[1], projection[1]], "k:")
    ax.scatter(*c, marker="*", s=260, color="black", zorder=6)
    ax.scatter(*point, color="#d62728", s=110, zorder=6)
    ax.scatter(*projection, marker="X", color="gold", edgecolor="black", s=180, zorder=6)
    ax.annotate("reference c", xy=c, xytext=(-65, 30), textcoords="offset points",
                arrowprops={"arrowstyle": "->"})
    ax.annotate("original point x", xy=point, xytext=(10, 10), textcoords="offset points")
    ax.annotate("projected point p", xy=projection, xytext=(15, 25),
                textcoords="offset points", arrowprops={"arrowstyle": "->"})
    ax.text(0.65, 1.65, "$x-c$", color="#d62728", fontsize=14)
    ax.text(1.25, 0.2, "$p-c$", color="#2ca02c", fontsize=14)
    ax.set(xlim=(-4, 4), ylim=(-3, 4),
           title="Projection keeps the part of x - c along the line")
    ax.set_aspect("equal", adjustable="box")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in axes:
        ax.plot(line[:, 0], line[:, 1], "k-", lw=1.2)
        ax.scatter(*c, marker="*", s=220, color="black")
        ax.set(xlim=(-4, 4), ylim=(-3, 4), xlabel="$x_1$", ylabel="$x_2$")
        ax.set_aspect("equal", adjustable="box")
    scatter_classes(axes[0], X0, X1, alpha=0.5)
    axes[0].set_title("Before projection")
    scatter_classes(axes[1], P0, P1, alpha=0.7)
    axes[1].set_title("After projection: every point is on the line")
    fig.tight_layout()
    plt.show()


def show_projected_mean_variance(X0, X1, c, v, t0, t1, m0, m1):
    X0_relative = X0 - c
    X1_relative = X1 - c
    m0_relative = m0 - c
    m1_relative = m1 - c
    projected_m0 = t0.mean() * v
    projected_m1 = t1.mean() * v
    line = np.linspace(-4, 4, 2)[:, None] * v

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    scatter_classes(axes[0], X0_relative, X1_relative, alpha=0.3)
    axes[0].plot(line[:, 0], line[:, 1], "k-", lw=1.5)
    axes[0].scatter(0, 0, marker="*", s=220, color="black", label="reference c maps to 0")
    for mean, projected, color in [
        (m0_relative, projected_m0, "#1f77b4"),
        (m1_relative, projected_m1, "#ff7f0e"),
    ]:
        axes[0].plot([mean[0], projected[0]], [mean[1], projected[1]], ":", color=color)
        axes[0].scatter(*mean, marker="*", s=180, color=color, edgecolor="black")
        axes[0].scatter(*projected, marker="X", s=150, color=color, edgecolor="black")
    axes[0].set(xlim=(-4, 4), ylim=(-3, 4), ylabel="$x_2-c_2$",
                title="Centered 2D coordinates: horizontal position equals score t")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()

    bins = np.linspace(-4, 4, 28)
    axes[1].hist(t0, bins=bins, alpha=0.5, label="class 0")
    axes[1].hist(t1, bins=bins, alpha=0.5, label="class 1")
    for scores, color in [(t0, "#1f77b4"), (t1, "#ff7f0e")]:
        center = scores.mean()
        spread = scores.std(ddof=1)
        axes[1].axvline(center, color=color, lw=2)
        axes[1].axvspan(center - spread, center + spread, color=color, alpha=0.12)
    axes[1].set(xlim=(-4, 4), xlabel="projected score t (same scale as above)", ylabel="count",
                title="Means and ±1 standard deviation on the 1D score line")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def show_direction_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.linspace(-4, 4, 30)
    for ax, (name, scores0, scores1, J) in zip(axes, results):
        ax.hist(scores0, bins=bins, alpha=0.55, label="class 0")
        ax.hist(scores1, bins=bins, alpha=0.55, label="class 1")
        ax.axvline(scores0.mean(), color="#1f77b4", lw=2)
        ax.axvline(scores1.mean(), color="#ff7f0e", lw=2)
        ax.set_title(f"{name}\nJ = {J:.4f}")
        ax.set_xlabel("projected score")
    axes[0].set_ylabel("count")
    axes[0].legend()
    fig.tight_layout()
    plt.show()


def show_whitening(X0, X1, Z0, Z1, m0, m1, c, w, z_m0, z_m1, u_z):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_classes(axes[0], X0, X1, alpha=0.4)
    axes[0].plot([m0[0], m1[0]], [m0[1], m1[1]], "k--", label="mean difference")
    fisher_line = c + np.linspace(-4, 4, 2)[:, None] * w
    axes[0].plot(fisher_line[:, 0], fisher_line[:, 1], color="#2ca02c", lw=2.5,
                 label="Fisher direction")
    axes[0].scatter(*c, marker="*", s=220, color="black")
    axes[0].set(xlim=(-4, 4), ylim=(-3, 4), title="Original space: tilted spread")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()

    scatter_classes(axes[1], Z0, Z1, alpha=0.5)
    axes[1].set(xlabel="$z_1$", ylabel="$z_2$")
    z_mid = (z_m0 + z_m1) / 2
    z_bound = np.max(np.abs(np.vstack([Z0, Z1]))) * 1.15
    z_line = z_mid + np.linspace(-z_bound, z_bound, 2)[:, None] * u_z
    axes[1].plot(z_line[:, 0], z_line[:, 1], color="#2ca02c", lw=2.5,
                 label="whitened mean direction")
    axes[1].annotate("", xy=z_m1, xytext=z_m0,
                     arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 2.5})
    axes[1].scatter(0, 0, marker="*", s=220, color="black", label="mapped reference")
    axes[1].set(xlim=(-z_bound, z_bound), ylim=(-z_bound, z_bound),
                title="Whitened space: round within-class scatter")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def show_decision_boundary(X0_test, X1_test, c, w, accuracy):
    boundary_direction = np.array([-w[1], w[0]])
    boundary = c + np.linspace(-4, 4, 2)[:, None] * boundary_direction
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X0_test, X1_test, alpha=0.55)
    ax.plot(boundary[:, 0], boundary[:, 1], "k-", lw=2, label="decision boundary")
    ax.annotate("", xy=c + 1.5 * w, xytext=c,
                arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 3})
    ax.scatter(*c, marker="*", s=220, color="black")
    ax.text(*(c + 1.6 * w), "projection direction", color="#2ca02c")
    ax.set(xlim=(-4, 4), ylim=(-3, 4),
           title=f"LDA on fresh data: accuracy = {accuracy:.1%}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.show()


def show_iris_lda(X, y, Z, separation_ratio, target_names):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, name in enumerate(target_names):
        mask = y == k
        axes[0].scatter(X[mask, 2], X[mask, 3], s=28, alpha=0.7,
                        color=colors[k], label=name)
        axes[1].scatter(Z[mask, 0], Z[mask, 1], s=28, alpha=0.7,
                        color=colors[k], label=name)
    axes[0].set(xlabel="petal length (cm)", ylabel="petal width (cm)",
                title="Iris: two original features")
    axes[1].set(xlabel=f"LD1 ({separation_ratio[0]:.1%} of separation)",
                ylabel=f"LD2 ({separation_ratio[1]:.1%} of separation)",
                title="Iris: two Fisher discriminant scores")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    plt.show()