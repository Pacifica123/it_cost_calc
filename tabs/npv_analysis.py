# tabs/npv_tab.py
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import scrolledtext


def calculate_npv(investment, discount_rate, cash_flows):
    npv = -investment
    for t, cf in enumerate(cash_flows, start=1):
        npv += cf / ((1 + discount_rate) ** t)
    return npv


class NPVTab(ttk.Frame):
    def __init__(self, parent, crud=None):
        super().__init__(parent)
        self.crud = crud

        # Общий контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=15, pady=15)

        # Левая панель — ввод данных и результат
        left_frame = ttk.Labelframe(main_frame, text="Расчет NPV", bootstyle=INFO)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        # Правая панель — график
        right_frame = ttk.Labelframe(main_frame, text="График NPV", bootstyle=INFO)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # --- Ввод данных ---
        ttk.Label(left_frame, text="Начальные инвестиции (I):", font=("Segoe UI", 10, "bold"),
                  foreground="black").pack(anchor=W, pady=(5, 0), padx=10)
        self.invest_entry = ttk.Entry(left_frame, width=25)
        self.invest_entry.pack(pady=5, padx=10, fill=X)

        ttk.Label(left_frame, text="Денежные потоки по годам (через запятую):",
                  font=("Segoe UI", 10, "bold"), foreground="black").pack(anchor=W, pady=(5, 0), padx=10)
        self.cf_entry = ttk.Entry(left_frame, width=40)
        self.cf_entry.pack(pady=5, padx=10, fill=X)

        ttk.Label(left_frame, text="Ставка дисконтирования r (например 0.1):",
                  font=("Segoe UI", 10, "bold"), foreground="black").pack(anchor=W, pady=(5, 0), padx=10)
        self.r_entry = ttk.Entry(left_frame, width=25)
        self.r_entry.pack(pady=5, padx=10, fill=X)

        ttk.Button(left_frame, text="📊 Рассчитать NPV", bootstyle=SUCCESS,
                   command=self.calculate_and_plot).pack(pady=10, padx=10, fill=X)

        # --- Результат расчета ---
        self.result_text = scrolledtext.ScrolledText(left_frame, height=18, width=60, font=("Consolas", 9))
        self.result_text.pack(padx=10, pady=(5, 10), fill=BOTH, expand=True)

        # --- График ---
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.ax.set_title("График NPV", fontsize=10)
        self.ax.set_xlabel("Год")
        self.ax.set_ylabel("Накопленный NPV")

    def calculate_and_plot(self):
        try:
            investment = float(self.invest_entry.get())
            cash_flows = [float(x.strip()) for x in self.cf_entry.get().split(",")]
            r = float(self.r_entry.get())

            # --- Расчет ---
            debug_lines = []
            debug_lines.append(f"{'Год':>3} | {'I':>10} | {'CFt':>10} | {'DiscF':>10} | {'PVt':>10} | {'NPVt':>10}")
            debug_lines.append("-" * 70)

            total_npv = -investment
            accumulated_npv = [total_npv]
            total_pv = 0

            debug_lines.append(f"{1:>3} | {investment:>10.2f} | {-investment:>10.2f} | {'-':>10} | {0:>10.2f} | {total_npv:>10.2f}")

            for t, cf in enumerate(cash_flows, start=1):
                disc_factor = 1 / ((1 + r) ** t)
                PV_t = cf * disc_factor
                total_pv += cf
                total_npv += PV_t
                accumulated_npv.append(total_npv)
                debug_lines.append(f"{t + 1:>3} | {0:>10.2f} | {cf:>10.2f} | {disc_factor:>10.4f} | {PV_t:>10.2f} | {total_npv:>10.2f}")

            debug_lines.append("-" * 70)
            debug_lines.append(f"{'Сумма':>3} | {investment:>10.2f} | {'':>10} | {'':>10} | {total_pv:>10.2f} | {total_npv:>10.2f}")

            self.result_text.delete('1.0', 'end')
            self.result_text.insert('end', "\n".join(debug_lines))

            # --- Построение графика ---
            self.ax.clear()
            self.ax.plot(range(1, len(accumulated_npv) + 1), accumulated_npv,
                         marker='o', linestyle='-', color='#2E86C1', linewidth=2, markersize=6)
            self.ax.axhline(0, color='red', linestyle='--', linewidth=1)
            self.ax.set_xlabel("Год", fontsize=9)
            self.ax.set_ylabel("Накопленный NPV", fontsize=9)
            self.ax.set_title(f"NPV по годам (r = {r:.2f})", fontsize=10)
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.result_text.delete('1.0', 'end')
            self.result_text.insert('end', f"Ошибка: {e}")
