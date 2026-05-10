from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_route_metrics(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --route-metrics spec: {spec}")
    route, path = spec.split("=", 1)
    return route, path


def load_feature_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_route_rows(path: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["name"]
            out[name] = {key: float(value) for key, value in row.items() if key != "name" and value != ""}
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--route-metrics", action="append", required=True, help="route_id=metrics_csv_path")
    parser.add_argument("--base-route", required=True)
    parser.add_argument("--val-label", required=True)
    parser.add_argument("--test-label", required=True)
    parser.add_argument("--metric", default="proxy")
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--min-cluster-support", type=int, default=8)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_feature_rows(args.feature_csv)
    route_tables = {route: load_route_rows(path) for route, path in (parse_route_metrics(spec) for spec in args.route_metrics)}
    routes = list(route_tables)
    if args.base_route not in route_tables:
        raise ValueError(f"Base route {args.base_route} missing from route metrics.")

    val_rows = [row for row in rows if row["dataset"] == args.val_label]
    test_rows = [row for row in rows if row["dataset"] == args.test_label]
    if not val_rows or not test_rows:
        raise ValueError("Need both val and test rows for route analysis.")

    missing = [(route, row["name"]) for route, table in route_tables.items() for row in val_rows if row["name"] not in table]
    if missing:
        raise ValueError(f"Missing route metrics for val names, first missing={missing[0]}")

    val_by_cluster: dict[int, list[dict[str, str]]] = defaultdict(list)
    test_by_cluster: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in val_rows:
        val_by_cluster[int(row[args.cluster_column])].append(row)
    for row in test_rows:
        test_by_cluster[int(row[args.cluster_column])].append(row)

    cluster_rows: list[dict[str, object]] = []
    selected_route_by_cluster: dict[int, str] = {}

    for cluster in sorted(set(val_by_cluster) | set(test_by_cluster)):
        val_members = val_by_cluster.get(cluster, [])
        test_members = test_by_cluster.get(cluster, [])
        base_mean = sum(route_tables[args.base_route][row["name"]][args.metric] for row in val_members) / max(len(val_members), 1)
        route_means = {
            route: sum(route_tables[route][row["name"]][args.metric] for row in val_members) / max(len(val_members), 1)
            for route in routes
        }
        best_route = max(route_means, key=route_means.get)
        best_mean = route_means[best_route]
        selected_route = args.base_route
        if len(val_members) >= args.min_cluster_support and best_mean - base_mean >= args.min_gain:
            selected_route = best_route
        selected_route_by_cluster[cluster] = selected_route
        cluster_rows.append(
            {
                "cluster": cluster,
                "val_count": len(val_members),
                "test_count": len(test_members),
                "base_route": args.base_route,
                "base_mean": base_mean,
                "best_route": best_route,
                "best_mean": best_mean,
                "selected_route": selected_route,
                "selected_mean": route_means[selected_route],
                "gain_vs_base": route_means[selected_route] - base_mean,
            }
        )

    base_total = sum(route_tables[args.base_route][row["name"]][args.metric] for row in val_rows) / len(val_rows)
    selected_total = (
        sum(route_tables[selected_route_by_cluster[int(row[args.cluster_column])]][row["name"]][args.metric] for row in val_rows)
        / len(val_rows)
    )
    oracle_total = sum(max(route_tables[route][row["name"]][args.metric] for route in routes) for row in val_rows) / len(val_rows)

    test_name_rows: list[dict[str, object]] = []
    route_to_test_names: dict[str, list[str]] = defaultdict(list)
    for row in sorted(test_rows, key=lambda item: item["name"]):
        cluster = int(row[args.cluster_column])
        route = selected_route_by_cluster.get(cluster, args.base_route)
        route_to_test_names[route].append(row["name"])
        test_name_rows.append({"name": row["name"], "cluster": cluster, "selected_route": route})

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "cluster_route_summary.csv",
        [
            "cluster",
            "val_count",
            "test_count",
            "base_route",
            "base_mean",
            "best_route",
            "best_mean",
            "selected_route",
            "selected_mean",
            "gain_vs_base",
        ],
        cluster_rows,
    )
    write_csv(out_dir / "test_route_assignments.csv", ["name", "cluster", "selected_route"], test_name_rows)
    for route, names in route_to_test_names.items():
        path = out_dir / "test_names_by_route" / f"{route}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")

    print(f"base_mean={base_total:.6f}")
    print(f"selected_mean={selected_total:.6f}")
    print(f"oracle_mean={oracle_total:.6f}")
    print(f"selected_gain={selected_total - base_total:.6f}")
    print(f"wrote={out_dir}")


if __name__ == "__main__":
    main()
