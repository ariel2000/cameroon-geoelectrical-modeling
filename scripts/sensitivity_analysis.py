import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs/sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalized_sensitivity(reference, modified):
    """
    Calcule une sensibilité relative normalisée.
    """
    reference = np.asarray(reference)
    modified = np.asarray(modified)

    return np.linalg.norm(modified - reference) / np.linalg.norm(reference)


def synthetic_dc_response(depth, layer_resistivity, layer_thickness, target_size):
    """
    Indicateur simplifié de sensibilité DC.
    Plus la cible est profonde, plus l'effet diminue.
    """
    contrast = abs(layer_resistivity - 10) / layer_resistivity
    attenuation = np.exp(-depth / 120)
    size_effect = target_size / 80
    layer_effect = np.exp(-layer_thickness / 80)

    return contrast * attenuation * size_effect * layer_effect


def synthetic_ip_response(depth, layer_thickness, target_size):
    """
    Indicateur simplifié de sensibilité IP.
    L'IP reste plus sensible à une cible polarisable.
    """
    attenuation = np.exp(-depth / 160)
    size_effect = target_size / 80
    layer_effect = np.exp(-layer_thickness / 120)

    return 1.8 * attenuation * size_effect * layer_effect


def synthetic_tdem_response(depth, layer_resistivity, target_size):
    """
    Indicateur simplifié de sensibilité TDEM.
    Le TDEM est sensible aux conducteurs profonds, mais dépend fortement
    de la conductivité globale du milieu.
    """
    conductivity_effect = 1 / layer_resistivity
    attenuation = np.exp(-depth / 200)
    size_effect = target_size / 80

    return 100 * conductivity_effect * attenuation * size_effect


def plot_sensitivity(parameter_name, values, dc, ip, tdem, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(values, dc, marker="o", label="DC")
    plt.plot(values, ip, marker="s", label="IP")
    plt.plot(values, tdem, marker="^", label="TDEM")

    plt.xlabel(parameter_name)
    plt.ylabel("Indice de sensibilité normalisé")
    plt.title(f"Analyse de sensibilité : {parameter_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main():
    # Paramètres de référence
    ref_layer_thickness = 20
    ref_layer_resistivity = 80
    ref_target_depth = 100
    ref_target_size = 80

    # 1. Influence de l'épaisseur de la couche superficielle
    thickness_values = np.array([10, 20, 30, 40, 60, 80])

    dc = [
        synthetic_dc_response(ref_target_depth, ref_layer_resistivity, h, ref_target_size)
        for h in thickness_values
    ]
    ip = [
        synthetic_ip_response(ref_target_depth, h, ref_target_size)
        for h in thickness_values
    ]
    tdem = [
        synthetic_tdem_response(ref_target_depth, ref_layer_resistivity, ref_target_size)
        for h in thickness_values
    ]

    plot_sensitivity(
        "Épaisseur de la couche superficielle (m)",
        thickness_values,
        dc,
        ip,
        tdem,
        "sensitivity_surface_layer_thickness.png",
    )

    # 2. Influence de la résistivité de la couche superficielle
    resistivity_values = np.array([30, 50, 80, 120, 200, 300])

    dc = [
        synthetic_dc_response(ref_target_depth, rho, ref_layer_thickness, ref_target_size)
        for rho in resistivity_values
    ]
    ip = [
        synthetic_ip_response(ref_target_depth, ref_layer_thickness, ref_target_size)
        for rho in resistivity_values
    ]
    tdem = [
        synthetic_tdem_response(ref_target_depth, rho, ref_target_size)
        for rho in resistivity_values
    ]

    plot_sensitivity(
        "Résistivité de la couche superficielle (Ω·m)",
        resistivity_values,
        dc,
        ip,
        tdem,
        "sensitivity_surface_layer_resistivity.png",
    )

    # 3. Influence de la profondeur de la cible
    depth_values = np.array([50, 75, 100, 125, 150, 200])

    dc = [
        synthetic_dc_response(d, ref_layer_resistivity, ref_layer_thickness, ref_target_size)
        for d in depth_values
    ]
    ip = [
        synthetic_ip_response(d, ref_layer_thickness, ref_target_size)
        for d in depth_values
    ]
    tdem = [
        synthetic_tdem_response(d, ref_layer_resistivity, ref_target_size)
        for d in depth_values
    ]

    plot_sensitivity(
        "Profondeur de la cible (m)",
        depth_values,
        dc,
        ip,
        tdem,
        "sensitivity_target_depth.png",
    )

    # 4. Influence de la taille de la cible
    size_values = np.array([20, 40, 60, 80, 100, 120])

    dc = [
        synthetic_dc_response(ref_target_depth, ref_layer_resistivity, ref_layer_thickness, s)
        for s in size_values
    ]
    ip = [
        synthetic_ip_response(ref_target_depth, ref_layer_thickness, s)
        for s in size_values
    ]
    tdem = [
        synthetic_tdem_response(ref_target_depth, ref_layer_resistivity, s)
        for s in size_values
    ]

    plot_sensitivity(
        "Taille de la cible (m)",
        size_values,
        dc,
        ip,
        tdem,
        "sensitivity_target_size.png",
    )

    print("Analyse de sensibilité terminée.")
    print(f"Figures enregistrées dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
