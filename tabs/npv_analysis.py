import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Можно использовать отдельную функцию для расчета итогового NPV
def calculate_npv(investment, discount_rate, cash_flows):
    npv = -investment
    for t, cf in enumerate(cash_flows, start=1):
        npv += cf / ((1 + discount_rate) ** t)
    return npv

class NPVTab(tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud

        # Поля ввода
        tk.Label(self, text="Начальные инвестиции (I):").grid(row=0, column=0, sticky="w")
        self.invest_entry = tk.Entry(self)
        self.invest_entry.grid(row=0, column=1)

        tk.Label(self, text="Денежные потоки по годам (через запятую):").grid(row=1, column=0, sticky="w")
        self.cf_entry = tk.Entry(self, width=50)
        self.cf_entry.grid(row=1, column=1)

        tk.Label(self, text="Ставка дисконтирования r (например 0.1):").grid(row=2, column=0, sticky="w")
        self.r_entry = tk.Entry(self)
        self.r_entry.grid(row=2, column=1)

        # Кнопка расчета
        self.calc_button = tk.Button(self, text="Рассчитать NPV", command=self.calculate_and_plot)
        self.calc_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Поле для текста результата
        self.result_text = tk.Text(self, height=15, width=80)
        self.result_text.grid(row=4, column=0, columnspan=2)

        # Поле для графика
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

    def calculate_and_plot(self):
        try:
            investment = float(self.invest_entry.get())
            cash_flows = [float(x) for x in self.cf_entry.get().split(",")]
            r = float(self.r_entry.get())

            debug_lines = []
            debug_lines.append(f"{'Год':>3} | {'I':>10} | {'CFt':>10} | {'DiscF':>10} | {'PVt':>10} | {'NPVt':>10}")
            debug_lines.append("-"*70)

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
                debug_lines.append(f"{t+1:>3} | {0:>10.2f} | {cf:>10.2f} | {disc_factor:>10.4f} | {PV_t:>10.2f} | {total_npv:>10.2f}")

            debug_lines.append("-"*70)
            debug_lines.append(f"{'Сумма':>3} | {investment:>10.2f} | {'':>10} | {'':>10} | {total_pv:>10.2f} | {total_npv:>10.2f}")

            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "\n".join(debug_lines))

            self.ax.clear()
            self.ax.plot(range(1, len(accumulated_npv)+1), accumulated_npv, marker='o')
            self.ax.axhline(0, color='red', linestyle='--')
            self.ax.set_xlabel("Год")
            self.ax.set_ylabel("Накопленный NPV")
            self.ax.set_title("NPV по годам при ставке r")
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"Ошибка: {e}")

