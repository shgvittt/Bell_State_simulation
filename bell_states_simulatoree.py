import numpy as np
import matplotlib.pyplot as plt

# Basis order: |00>, |01>, |10>, |11>
I = np.array([[1, 0], [0, 1]], dtype=complex)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

X = np.array([[0, 1], [1, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

initial_state = np.array([1, 0, 0, 0], dtype=complex)

OUTCOMES = ["|00>", "|01>", "|10>", "|11>"]
PROBABILITY_TOLERANCE = 1e-10


# Create the four Bell states
def create_bell_states():
    after_h = np.kron(H, I) @ initial_state

    phi_plus = CNOT @ after_h
    phi_minus = np.kron(Z, I) @ phi_plus
    psi_plus = np.kron(I, X) @ phi_plus
    psi_minus = np.kron(Z, I) @ psi_plus

    return {
        "1": ("Phi+", phi_plus),
        "2": ("Phi-", phi_minus),
        "3": ("Psi+", psi_plus),
        "4": ("Psi-", psi_minus)
    }


# Format amplitudes, including complex numbers
def display_amplitude(amplitude):
    real = float(np.real(amplitude))
    imag = float(np.imag(amplitude))

    if np.isclose(imag, 0.0):
        return f"{real:.4f}"

    if np.isclose(real, 0.0):
        return f"{imag:.4f}j"

    sign = "+" if imag >= 0 else "-"

    return f"{real:.4f}{sign}{abs(imag):.4f}j"


# Display the quantum state
def display_quantum_state(name, state):
    print(f"\nSelected State: {name}")

    print("\nState vector:")
    print(np.array2string(np.round(state, 4)))

    norm = np.linalg.norm(state)

    print(f"State norm: {norm:.4f}")

    print(
        "Normalization check: "
        + ("Passed" if np.isclose(norm, 1.0, atol=1e-8)
           else "Failed")
    )

    terms = [
        f"({display_amplitude(a)}){o}"
        for a, o in zip(state, OUTCOMES)
        if not np.isclose(a, 0.0)
    ]

    print("State notation:")
    print(" + ".join(terms) if terms else "Zero state")


# Calculate theoretical measurement probabilities
def calculate_probabilities(state, show=True):
    state = np.asarray(state, dtype=complex)

    if state.shape != (4,):
        raise ValueError(
            "A two-qubit state must contain exactly four amplitudes."
        )

    if not np.all(np.isfinite(state)):
        raise ValueError("State amplitudes must be finite numbers.")

    norm = np.linalg.norm(state)

    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError(
            f"State is not normalized (norm={norm:.8f}); "
            "measurement stopped."
        )

    probabilities = np.abs(state) ** 2
    total = float(np.sum(probabilities))

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(
            total, 1.0, atol=PROBABILITY_TOLERANCE
        )
    ):
        raise ValueError(
            "Measurement probabilities must be valid and sum to 1."
        )

    probabilities = probabilities / total

    if show:
        print("\nTheoretical measurement probabilities:")

        for outcome, p in zip(OUTCOMES, probabilities):
            print(f"{outcome}: {p:.4f} ({p * 100:.2f}%)")

    return probabilities


# Simulate repeated measurements
def perform_measurement_shots(probabilities, shots):
    if (
        isinstance(shots, bool)
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
    ):
        raise ValueError(
            "The number of shots must be a positive integer."
        )

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (4,):
        raise ValueError(
            "Exactly four measurement probabilities are required."
        )

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(
            np.sum(probabilities),
            1.0,
            atol=PROBABILITY_TOLERANCE
        )
    ):
        raise ValueError(
            "Probabilities must be finite, non-negative, and sum to 1."
        )

    probabilities = probabilities / np.sum(probabilities)

    measurements = np.random.choice(
        OUTCOMES,
        size=shots,
        p=probabilities
    )

    counts = {}
    percentages = {}

    print("\nRepeated measurement results:")
    print(f"Total shots: {shots}")

    for outcome in OUTCOMES:
        counts[outcome] = int(
            np.count_nonzero(measurements == outcome)
        )

        percentages[outcome] = counts[outcome] / shots * 100

        print(
            f"{outcome}: {counts[outcome]} counts "
            f"({percentages[outcome]:.2f}%)"
        )

    return counts, percentages


# Explain differences between theoretical and measured results
def explain_probability_difference():
    print("\nWhy do theoretical and measured probabilities differ?")
    print("-" * 55)

    print(
        "Theoretical probabilities are calculated from the state vector."
    )

    print(
        "Measured percentages come from randomly simulated finite shots."
    )

    print(
        "Random sampling causes statistical fluctuations, "
        "so the values may differ."
    )

    print(
        "More shots generally reduce fluctuations. "
        "Zero-probability outcomes never appear in this ideal simulator."
    )

    print("-" * 55)


# Plot theoretical and measured probabilities
def plot_results(probabilities, percentages, state_name):
    theoretical = np.asarray(probabilities, dtype=float) * 100

    measured = [percentages[o] for o in OUTCOMES]

    x = np.arange(len(OUTCOMES))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        theoretical,
        width,
        label="Theoretical probability (%)"
    )

    plt.bar(
        x + width / 2,
        measured,
        width,
        label="Measured percentage (%)"
    )

    plt.xticks(x, OUTCOMES)
    plt.xlabel("Computational basis outcome")
    plt.ylabel("Probability / percentage (%)")

    plt.title(
        f"{state_name}: Theoretical vs Measured Results"
    )

    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Get a valid positive integer from the user
def get_positive_integer():
    while True:
        user_input = input(
            "\nEnter the number of measurement shots: "
        ).strip()

        try:
            shots = int(user_input)

            if shots > 0:
                return shots

            print(
                "Please enter a positive integer greater than zero."
            )

        except ValueError:
            print(
                "Invalid input. Enter a whole number, such as 1000."
            )


# Run the complete measurement process
def handle_measurement(name, state):
    probabilities = calculate_probabilities(state)

    shots = get_positive_integer()

    counts, percentages = perform_measurement_shots(
        probabilities, shots
    )

    explain_probability_difference()

    plot_results(probabilities, percentages, name)

    return counts, percentages


# Recognize whether a state matches one of the four Bell states
def recognize_bell_state(state, bell_states):
    state = np.asarray(state, dtype=complex)

    if state.shape != (4,) or not np.all(np.isfinite(state)):
        return None

    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-8):
        return None

    for name, bell_state in bell_states.values():
        overlap = np.vdot(bell_state, state)

        if np.isclose(abs(overlap), 1.0, atol=1e-8):
            phase = overlap / abs(overlap)

            if np.allclose(
                state, phase * bell_state, atol=1e-8
            ):
                return name

    return None


# Build a quantum circuit step by step
def circuit_builder(bell_states):
    state = initial_state.copy()

    while True:
        print("\n--- QUANTUM CIRCUIT BUILDER ---")

        print(
            "Current state:",
            np.array2string(np.round(state, 4))
        )

        print("\nChoose an operation:")
        print("1. Apply H to first qubit")
        print("2. Apply X to first qubit")
        print("3. Apply Z to first qubit")
        print("4. Apply H to second qubit")
        print("5. Apply X to second qubit")
        print("6. Apply Z to second qubit")
        print("7. Apply CNOT (first qubit controls second)")
        print("8. Show measurement probabilities")
        print("9. Reset state")
        print("10. Back to main menu")

        choice = input("Enter your choice (1-10): ").strip()

        if choice == "1":
            state = np.kron(H, I) @ state
            print("\nApplied H to the first qubit.")

        elif choice == "2":
            state = np.kron(X, I) @ state
            print("\nApplied X to the first qubit.")

        elif choice == "3":
            state = np.kron(Z, I) @ state
            print("\nApplied Z to the first qubit.")

        elif choice == "4":
            state = np.kron(I, H) @ state
            print("\nApplied H to the second qubit.")

        elif choice == "5":
            state = np.kron(I, X) @ state
            print("\nApplied X to the second qubit.")

        elif choice == "6":
            state = np.kron(I, Z) @ state
            print("\nApplied Z to the second qubit.")

        elif choice == "7":
            state = CNOT @ state
            print("\nApplied CNOT.")

        elif choice == "8":
            try:
                calculate_probabilities(state)

            except ValueError as error:
                print(f"\nCannot show probabilities: {error}")

            continue

        elif choice == "9":
            state = initial_state.copy()

            print("\nState reset to |00>.")

            display_quantum_state("Custom State", state)

            continue

        elif choice == "10":
            print("\nReturning to the main menu.")
            return

        else:
            print(
                "Invalid choice. Please select an option from 1 to 10."
            )
            continue

        display_quantum_state("Custom State", state)

        bell_name = recognize_bell_state(state, bell_states)

        if bell_name is not None:
            print(f"\nBell state recognized: {bell_name}!")
            print(f"You created the Bell state {bell_name}!")

            print(
                "Let's measure it. The circuit state will remain unchanged."
            )

            try:
                handle_measurement(bell_name, state)

            except ValueError as error:
                print(
                    f"\nMeasurement could not be completed: {error}"
                )

        else:
            print(
                "\nThis state is not one of the four Bell states."
            )


# Main menu
def main():
    bell_states = create_bell_states()

    while True:
        print("\n" + "=" * 30)
        print("    BELL STATES SIMULATOR")
        print("=" * 30)

        print("1. Phi+")
        print("2. Phi-")
        print("3. Psi+")
        print("4. Psi-")
        print("5. Build a Bell state step by step")
        print("6. Exit")

        choice = input("\nChoose an option (1-6): ").strip()

        if choice in ("1", "2", "3", "4"):
            name, state = bell_states[choice]

            display_quantum_state(name, state)

            try:
                handle_measurement(name, state)

            except ValueError as error:
                print(
                    f"\nMeasurement could not be completed: {error}"
                )

        elif choice == "5":
            circuit_builder(bell_states)

        elif choice == "6":
            print("Exiting Bell States Simulator. Goodbye!")
            break

        else:
            print(
                "Invalid choice. Please select an option from 1 to 6."
            )


if __name__ == "__main__":
    main()