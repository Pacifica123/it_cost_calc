"""NPV tab using application use case for report building."""

from __future__ import annotations

from tkinter import BOTH, LEFT, RIGHT, W, X, scrolledtext

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import INFO, SUCCESS

    _BOOTSTYLE_SUPPORTED = True
except ModuleNotFoundError:  # pragma: no cover - fallback for lean environments
    from tkinter import ttk

    INFO = ""
    SUCCESS = ""
    _BOOTSTYLE_SUPPORTED = False

from application.use_cases.build_npv_report import BuildNpvReportUseCase


def _bootstyle_kwargs(value: str) -> dict[str, str]:
    if _BOOTSTYLE_SUPPORTED and value:
        return {"bootstyle": value}
    return {}


class NPVTab(ttk.Frame):
    def __init__(self, parent, build_npv_report_use_case: BuildNpvReportUseCase):
        super().__init__(parent)
        self.build_npv_report_use_case = build_npv_report_use_case

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=15, pady=15)

        left_frame = ttk.Labelframe(
            main_frame,
            text="Расчет NPV",
            **_bootstyle_kwargs(INFO),
        )
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        right_frame = ttk.Labelframe(
            main_frame,
            text="График NPV",
            **_bootstyle_kwargs(INFO),
        )
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        ttk.Label(
            left_frame,
            text="Начальные инвестиции (I):",
            font=("Segoe UI", 10, "bold"),
            foreground="black",
        ).pack(anchor=W, pady=(5, 0), padx=10)
        self.invest_entry = ttk.Entry(left_frame, width=25)
        self.invest_entry.pack(pady=5, padx=10, fill=X)

        ttk.Label(
            left_frame,
            text="Денежные потоки по годам (через запятую):",
            font=("Segoe UI", 10, "bold"),
            foreground="black",
        ).pack(anchor=W, pady=(5, 0), padx=10)
        self.cf_entry = ttk.Entry(left_frame, width=40)
        self.cf_entry.pack(pady=5, padx=10, fill=X)

        ttk.Label(
            left_frame,
            text="Ставка дисконтирования r (например 0.1):",
            font=("Segoe UI", 10, "bold"),
            foreground="black",
        ).pack(anchor=W, pady=(5, 0), padx=10)
        self.r_entry = ttk.Entry(left_frame, width=25)
        self.r_entry.pack(pady=5, padx=10, fill=X)

        ttk.Button(
            left_frame,
            text="📊 Рассчитать NPV",
            command=self.calculate_and_plot,
            **_bootstyle_kwargs(SUCCESS),
        ).pack(pady=10, padx=10, fill=X)

        self.result_text = scrolledtext.ScrolledText(
            left_frame, height=18, width=60, font=("Consolas", 9)
        )
        self.result_text.pack(padx=10, pady=(5, 10), fill=BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.ax.set_title("График NPV", fontsize=10)
        self.ax.set_xlabel("Год")
        self.ax.set_ylabel("Накопленный NPV")

    def calculate_and_plot(self) -> None:
        try:
            investment = float(self.invest_entry.get())
            cash_flows = [float(x.strip()) for x in self.cf_entry.get().split(",") if x.strip()]
            discount_rate = float(self.r_entry.get())

            report = self.build_npv_report_use_case.execute(
                investment=investment,
                discount_rate=discount_rate,
                cash_flows=cash_flows,
            )

            debug_lines = [
                f"{'Год':>3} | {'I':>10} | {'CFt':>10} | {'DiscF':>10} | {'PVt':>10} | {'NPVt':>10}",
                "-" * 70,
            ]
            for row in report["rows"]:
                discount_factor = row["discount_factor"]
                discount_factor_text = "-" if discount_factor is None else f"{discount_factor:.4f}"
                debug_lines.append(
                    f"{row['year']:>3} | "
                    f"{row['investment']:>10.2f} | "
                    f"{row['cash_flow']:>10.2f} | "
                    f"{discount_factor_text:>10} | "
                    f"{row['present_value']:>10.2f} | "
                    f"{row['accumulated_npv']:>10.2f}"
                )
            debug_lines.append("-" * 70)
            debug_lines.append(f"Итоговый NPV: {report['npv']:.2f}")

            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", "\n".join(debug_lines))

            self.ax.clear()
            self.ax.plot(
                range(0, len(report["accumulated_points"])),
                report["accumulated_points"],
                marker="o",
                linestyle="-",
                color="#2E86C1",
                linewidth=2,
                markersize=6,
            )
            self.ax.axhline(0, color="red", linestyle="--", linewidth=1)
            self.ax.set_xlabel("Год", fontsize=9)
            self.ax.set_ylabel("Накопленный NPV", fontsize=9)
            self.ax.set_title(f"NPV по годам (r = {discount_rate:.2f})", fontsize=10)
            self.ax.grid(True, linestyle=":", alpha=0.6)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as error:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Ошибка: {error}")

    def shutdown(self) -> None:
        try:
            self.canvas.get_tk_widget().destroy()
        except Exception:
            pass
        plt.close(self.fig)
