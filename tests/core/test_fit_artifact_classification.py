"""Construct-level contracts for fit-artifact classification."""

from imas_codex.core.node_classifier import classify_node_pass1


def test_constraint_and_profile_weights_share_fit_artifact_category() -> None:
    """Equivalent fit weights receive one category across IDS structures."""
    paths = (
        "equilibrium/time_slice/constraints/faraday_angle/weight",
        "plasma_profiles/profiles_1d/ion/temperature_fit/weight",
    )

    categories = {
        classify_node_pass1(path, "weight", data_type="FLT_0D", unit="1")
        for path in paths
    }

    assert categories == {"fit_artifact"}


def test_constraint_reconstructed_values_are_fit_artifacts() -> None:
    """Reconstructed constraint values are fitting outputs, not quantities."""
    paths = (
        "equilibrium/time_slice/constraints/ip/reconstructed",
        "equilibrium/time_slice/constraints/flux_loop/flux_reconstructed",
    )

    for path in paths:
        leaf = path.rsplit("/", 1)[-1]
        assert (
            classify_node_pass1(path, leaf, data_type="FLT_0D", unit="Wb")
            == "fit_artifact"
        )


def test_convergence_iteration_counts_are_fit_artifacts_across_ids() -> None:
    """Solver iteration counts share one bookkeeping classification."""
    paths = (
        "equilibrium/time_slice/convergence/iterations_n",
        "transport_solver_numerics/solver_1d/equation/convergence/iterations_n",
    )

    for path in paths:
        assert (
            classify_node_pass1(path, "iterations_n", data_type="INT_0D")
            == "fit_artifact"
        )


def test_detector_humidity_remains_a_dimensionless_quantity() -> None:
    """A documented dimensionless detector measurement remains nameable."""
    path = "camera_x_rays/detector_humidity"
    assert (
        classify_node_pass1(path, "detector_humidity", data_type="FLT_0D", unit="1")
        == "quantity"
    )
