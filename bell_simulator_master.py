
"""
Master Bell States Simulator
Tkinter GUI + Matplotlib visualization using the updated quantum engine.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import bell_states_simulatoree as engine


# ==========================================
# CONNECT TO UPDATED QUANTUM ENGINE
# ==========================================

H = engine.H
I = engine.I
X = engine.X
Z = engine.Z
CNOT = engine.CNOT

INITIAL_STATE = engine.initial_state
OUTCOMES = engine.OUTCOMES

ENGINE_BELL_STATES = engine.create_bell_states()

BELL_STATES_DICT = {
    "Φ⁺ (|00⟩ + |11⟩) / √2": ENGINE_BELL_STATES["1"][1],
    "Φ⁻ (|00⟩ - |11⟩) / √2": ENGINE_BELL_STATES["2"][1],
    "Ψ⁺ (|01⟩ + |10⟩) / √2": ENGINE_BELL_STATES["3"][1],
    "Ψ⁻ (|01⟩ - |10⟩) / √2": ENGINE_BELL_STATES["4"][1],
}


def recognize_bell_state(state):
    return engine.recognize_bell_state(
        state, ENGINE_BELL_STATES
    )


def get_shots(variable):
    try:
        shots = int(variable.get())
        if shots > 0:
            return shots
    except ValueError:
        pass

    messagebox.showwarning(
        "Invalid Input",
        "Enter a positive whole number. Using 1000 shots."
    )
    variable.set("1000")
    return 1000


# ==========================================
# GUI APPLICATION
# ==========================================

class MasterBellSimulatorApp:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "Master Bell States Simulator — Linear Algebra & Visualization"
        )
        self.root.geometry("1250x820")
        self.root.minsize(980, 720)

        self.current_builder_state = INITIAL_STATE.copy()

        self.build_ui()
        self.run_preset_simulation()

    # --------------------------------------
    # MAIN INTERFACE
    # --------------------------------------

    def build_ui(self):
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="QUANTUM ENTANGLEMENT & BELL STATES",
            font=("Arial", 18, "bold")
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Updated NumPy engine, interactive state tracking, and live sampling.",
            font=("Arial", 10)
        ).pack(anchor="w", pady=(2, 4))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill="both", expand=True, padx=16, pady=(0, 8)
        )

        self.tab_presets = ttk.Frame(
            self.notebook, padding=10
        )
        self.tab_builder = ttk.Frame(
            self.notebook, padding=10
        )

        self.notebook.add(
            self.tab_presets, text="  1. Preset Bell States  "
        )
        self.notebook.add(
            self.tab_builder, text="  2. Step-by-Step Circuit Builder  "
        )

        self.notebook.bind(
            "<<NotebookTabChanged>>", self.on_tab_change
        )

        self.build_preset_tab()
        self.build_builder_tab()

    # --------------------------------------
    # PRESET BELL STATES TAB
    # --------------------------------------

    def build_preset_tab(self):
        frame = self.tab_presets

        controls = ttk.Frame(frame, padding=(0, 5, 0, 10))
        controls.pack(fill="x")

        ttk.Label(
            controls, text="Select Bell State:"
        ).pack(side="left", padx=(0, 8))

        self.preset_var = tk.StringVar(
            value=list(BELL_STATES_DICT.keys())[0]
        )

        selector = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=list(BELL_STATES_DICT.keys()),
            state="readonly",
            width=30
        )
        selector.pack(side="left", padx=(0, 15))
        selector.bind(
            "<<ComboboxSelected>>",
            lambda event: self.run_preset_simulation()
        )

        ttk.Label(
            controls, text="Shots:"
        ).pack(side="left", padx=(0, 5))

        self.preset_shots_var = tk.StringVar(value="1000")

        ttk.Entry(
            controls,
            textvariable=self.preset_shots_var,
            width=8
        ).pack(side="left", padx=(0, 12))

        ttk.Button(
            controls,
            text="Run Simulation",
            command=self.run_preset_simulation
        ).pack(side="left")

        paned = ttk.Panedwindow(
            frame, orient="horizontal"
        )
        paned.pack(fill="both", expand=True, pady=5)

        left = ttk.Frame(paned, padding=5)
        right = ttk.Frame(paned, padding=5)

        paned.add(left, weight=1)
        paned.add(right, weight=2)

        ttk.Label(
            left,
            text="MATHEMATICAL STATE VECTOR",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.preset_state_text = tk.Text(
            left,
            height=10,
            width=38,
            font=("Courier New", 10),
            state="disabled"
        )
        self.preset_state_text.pack(
            fill="x", pady=(5, 10)
        )

        ttk.Label(
            left,
            text="PHYSICAL INTERPRETATION",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.preset_info_text = tk.Text(
            left,
            height=16,
            width=42,
            wrap="word",
            font=("Arial", 10),
            relief="flat",
            background="#f4f4f4",
            padx=8,
            pady=8,
            state="disabled"
        )
        self.preset_info_text.pack(
            fill="both", expand=True, pady=(5, 0)
        )

        ttk.Label(
            right,
            text="MEASUREMENT PROBABILITIES (THEORY vs SAMPLED)",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.fig1 = Figure(
            figsize=(6, 4.5), dpi=100, tight_layout=True
        )
        self.ax1 = self.fig1.add_subplot(111)

        self.canvas1 = FigureCanvasTkAgg(
            self.fig1, master=right
        )
        self.canvas1.get_tk_widget().pack(
            fill="both", expand=True, pady=(5, 0)
        )

    # --------------------------------------
    # CIRCUIT BUILDER TAB
    # --------------------------------------

    def build_builder_tab(self):
        paned = ttk.Panedwindow(
            self.tab_builder, orient="horizontal"
        )
        paned.pack(fill="both", expand=True, pady=5)

        left = ttk.Frame(paned, padding=5)
        right = ttk.Frame(paned, padding=5)

        paned.add(left, weight=1)
        paned.add(right, weight=2)

        ttk.Label(
            left,
            text="APPLY QUANTUM GATES",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(0, 5))

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=5)

        gate_buttons = [
            ("Apply H to Qubit 0", "H0"),
            ("Apply X to Qubit 0", "X0"),
            ("Apply Z to Qubit 0", "Z0"),
            ("Apply H to Qubit 1", "H1"),
            ("Apply X to Qubit 1", "X1"),
            ("Apply Z to Qubit 1", "Z1"),
            ("Apply CNOT (Q0 ➔ Q1)", "CNOT")
        ]

        for label, gate in gate_buttons:
            ttk.Button(
                buttons,
                text=label,
                command=lambda g=gate: self.apply_gate(g)
            ).pack(fill="x", pady=2)

        ttk.Button(
            buttons,
            text="Reset State to |00⟩",
            command=self.reset_builder
        ).pack(fill="x", pady=(10, 2))

        ttk.Label(
            left,
            text="STATE VECTOR & RECOGNITION",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(8, 2))

        self.builder_state_text = tk.Text(
            left,
            height=8,
            width=38,
            font=("Courier New", 10),
            state="disabled"
        )
        self.builder_state_text.pack(
            fill="x", pady=(2, 0)
        )

        controls = ttk.Frame(
            right, padding=(0, 2, 0, 5)
        )
        controls.pack(fill="x")

        ttk.Label(
            controls, text="Shots:"
        ).pack(side="left", padx=(0, 5))

        self.builder_shots_var = tk.StringVar(value="1000")

        ttk.Entry(
            controls,
            textvariable=self.builder_shots_var,
            width=8
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            controls,
            text="Run Measurement & Plot",
            command=self.update_builder_display
        ).pack(side="left")

        ttk.Label(
            right,
            text="CIRCUIT MEASUREMENT (THEORY vs SAMPLED)",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(5, 0))

        self.fig2 = Figure(
            figsize=(6, 4.3), dpi=100, tight_layout=True
        )
        self.ax2 = self.fig2.add_subplot(111)

        self.canvas2 = FigureCanvasTkAgg(
            self.fig2, master=right
        )
        self.canvas2.get_tk_widget().pack(
            fill="both", expand=True, pady=(5, 0)
        )

    # --------------------------------------
    # HELPER METHODS
    # --------------------------------------

    def set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def format_state(self, state):
        lines = []

        for outcome, amplitude in zip(OUTCOMES, state):
            lines.append(
                f"{outcome} : "
                f"{amplitude.real:+.3f}"
                f"{amplitude.imag:+.3f}i"
            )

        return "\n".join(lines)

    def sample(self, state, shots):
        probabilities = engine.calculate_probabilities(
            state, show=False
        )

        counts, percentages = engine.perform_measurement_shots(
            probabilities, shots
        )

        measured = np.array(
            [percentages[o] for o in OUTCOMES],
            dtype=float
        )

        theoretical = probabilities * 100

        return theoretical, measured

    def draw_chart(self, ax, canvas, state, shots, title):
        theoretical, measured = self.sample(state, shots)

        ax.clear()

        x = np.arange(len(OUTCOMES))
        width = 0.35

        ax.bar(
            x - width / 2,
            theoretical,
            width,
            label="Theoretical (%)"
        )

        measured_bars = ax.bar(
            x + width / 2,
            measured,
            width,
            label=f"Measured ({shots} shots)"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(OUTCOMES)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 115)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        ax.bar_label(
            measured_bars,
            fmt="%.1f%%",
            padding=3,
            fontsize=9
        )

        canvas.draw()

    # --------------------------------------
    # PRESET SIMULATION
    # --------------------------------------

    def run_preset_simulation(self):
        label = self.preset_var.get()
        state = BELL_STATES_DICT[label]
        shots = get_shots(self.preset_shots_var)

        summary = (
            self.format_state(state)
            + f"\n\nNorm: {np.linalg.norm(state):.4f} (Valid)"
        )

        self.set_text(
            self.preset_state_text, summary
        )

        information = {
            "Φ⁺ (|00⟩ + |11⟩) / √2":
                "Entangled state: both qubits give the same result, "
                "00 or 11, with theoretical probabilities of 50% each.",

            "Φ⁻ (|00⟩ - |11⟩) / √2":
                "Entangled state: both qubits give the same result, "
                "00 or 11, with theoretical probabilities of 50% each. "
                "The minus sign represents a relative quantum phase.",

            "Ψ⁺ (|01⟩ + |10⟩) / √2":
                "Entangled state: the qubits give opposite results, "
                "01 or 10, with theoretical probabilities of 50% each.",

            "Ψ⁻ (|01⟩ - |10⟩) / √2":
                "Entangled state: the qubits give opposite results, "
                "01 or 10, with theoretical probabilities of 50% each. "
                "The minus sign represents a relative quantum phase."
        }

        explanation = (
            information[label]
            + "\n\nWhy aren't the results exactly 50%?\n"
            + "Each measurement shot randomly produces an outcome. "
            "With a finite number of shots, measured percentages "
            "may differ slightly from 50%. Increasing the number "
            "of shots generally brings them closer to the theoretical "
            "probabilities."
        )

        self.set_text(self.preset_info_text, explanation)

        try:
            self.draw_chart(
                self.ax1,
                self.canvas1,
                state,
                shots,
                f"State: {label.split()[0]} ({shots:,} shots)"
            )
        except ValueError as error:
            messagebox.showerror(
                "Measurement Error", str(error)
            )

    # --------------------------------------
    # CIRCUIT BUILDER DISPLAY
    # --------------------------------------

    def update_builder_display(self):
        state = self.current_builder_state
        bell_name = recognize_bell_state(state)

        if bell_name:
            status = f"\nRecognized Bell State: {bell_name}"
        else:
            status = "\nState: Superposition / Product state"

        self.set_text(
            self.builder_state_text,
            self.format_state(state) + status
        )

        shots = get_shots(self.builder_shots_var)

        title = "Custom Circuit Probabilities"

        if bell_name:
            title += f" [{bell_name}]"

        try:
            self.draw_chart(
                self.ax2,
                self.canvas2,
                state,
                shots,
                title
            )
        except ValueError as error:
            messagebox.showerror(
                "Measurement Error", str(error)
            )

    # --------------------------------------
    # APPLY GATES
    # --------------------------------------

    def apply_gate(self, gate_type):
        operators = {
            "H0": np.kron(H, I),
            "X0": np.kron(X, I),
            "Z0": np.kron(Z, I),
            "H1": np.kron(I, H),
            "X1": np.kron(I, X),
            "Z1": np.kron(I, Z),
            "CNOT": CNOT
        }

        if gate_type in operators:
            self.current_builder_state = (
                operators[gate_type]
                @ self.current_builder_state
            )

            self.update_builder_display()

    def reset_builder(self):
        self.current_builder_state = INITIAL_STATE.copy()
        self.update_builder_display()

    def on_tab_change(self, event):
        selected_tab = self.notebook.index(
            self.notebook.select()
        )

        if selected_tab == 1:
            self.update_builder_display()


# ==========================================
# RUN APPLICATION
# ==========================================

def main():
    root = tk.Tk()

    try:
        style = ttk.Style()

        if "clam" in style.theme_names():
            style.theme_use("clam")

    except tk.TclError:
        pass

    MasterBellSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
