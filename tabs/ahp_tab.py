# tabs/ahp_tab.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import math
import numpy as np

# Импортируем run_ahp_pipeline из optcore2 (папка должна содержать __init__.py)
try:
    from optcore2.ahp_config_selector import run_ahp_pipeline, aggregate_configuration
except Exception:
    # Если по каким-то причинам импорт не проходит — будем работать локально, но интерфейс всё равно доступен.
    run_ahp_pipeline = None
    aggregate_configuration = None

DEFAULT_SOFT_CRITERIA = ['avg_reliability', 'total_performance', 'total_cost', 'total_energy', 'lifespan']

# Встроенные матрицы экспертов (используются, т.к. галка доверия неактивна и всегда включена)
M1 = np.array([
    [1, 3, 5, 4, 3],
    [1/3, 1, 3, 2, 2],
    [1/5, 1/3, 1, 1/2, 1/3],
    [1/4, 1/2, 2, 1, 1/2],
    [1/3, 1/2, 3, 2, 1],
], dtype=float)
M2 = np.array([
    [1, 2, 4, 3, 2],
    [1/2, 1, 2, 2, 1],
    [1/4, 1/2, 1, 1/2, 1/3],
    [1/3, 1/2, 2, 1, 1/2],
    [1/2, 1, 3, 2, 1],
], dtype=float)


class AHPAnalysisTab(tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud

        # Внутренние данные: словарь конфигураций id -> {'devices': [...], 'meta': {...}}
        self.configurations = {}

        # Верх: панель управления конфигурациями
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=6)

        tk.Button(ctrl, text="Добавить конфигурацию", command=self._add_config_popup).pack(side="left")
        tk.Button(ctrl, text="Удалить выбранную", command=self._delete_selected_config).pack(side="left")
        tk.Button(ctrl, text="Загрузить из JSON...", command=self._load_from_json).pack(side="left")
        tk.Button(ctrl, text="Сохранить конфигурации в JSON...", command=self._save_configs_to_json).pack(side="left")

        # Таблица конфигураций
        self.cfg_table = ttk.Treeview(self, columns=("id", "people", "devices_count"), show="headings", height=6)
        self.cfg_table.heading("id", text="ID")
        self.cfg_table.heading("people", text="People")
        self.cfg_table.heading("devices_count", text="Devices")
        self.cfg_table.pack(fill="x", padx=6, pady=(0,6))
        self.cfg_table.bind("<<TreeviewSelect>>", self._on_select_config)

        # Секция — список устройств выбранной конфигурации
        dev_frame = tk.Frame(self)
        dev_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(dev_frame, text="Устройства выбранной конфигурации:").pack(anchor="w")
        self.dev_table = ttk.Treeview(dev_frame, columns=("role","vendor","cpu","ram","energy","cost","rel_low","rel_high"), show="headings", height=6)
        for col, name in [("role","Role"),("vendor","Vendor"),("cpu","CPU"),("ram","RAM"),("energy","Energy"),("cost","Cost"),("rel_low","Rel low"),("rel_high","Rel high")]:
            self.dev_table.heading(col, text=name)
        self.dev_table.pack(fill="x")
        # Кнопки для устройств
        dev_btns = tk.Frame(self)
        dev_btns.pack(fill="x", padx=6)
        tk.Button(dev_btns, text="Добавить устройство", command=self._add_device_popup).pack(side="left")
        tk.Button(dev_btns, text="Удалить устройство", command=self._delete_device).pack(side="left")

        # Constraints и запуск
        conf_frame = tk.LabelFrame(self, text="Параметры анализа")
        conf_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(conf_frame, text="Max budget:").grid(row=0, column=0, sticky="w")
        self.e_budget = tk.Entry(conf_frame, width=12); self.e_budget.grid(row=0, column=1, sticky="w"); self.e_budget.insert(0, "6000")
        tk.Label(conf_frame, text="Max energy:").grid(row=0, column=2, sticky="w")
        self.e_energy = tk.Entry(conf_frame, width=12); self.e_energy.grid(row=0, column=3, sticky="w"); self.e_energy.insert(0, "700")
        tk.Label(conf_frame, text="People tolerance (0..1):").grid(row=0, column=4, sticky="w")
        self.e_tol = tk.Entry(conf_frame, width=8); self.e_tol.grid(row=0, column=5, sticky="w"); self.e_tol.insert(0, "0.25")

        # Soft criteria editable field (comma-separated) — по умолчанию DEFAULT_SOFT_CRITERIA
        tk.Label(conf_frame, text="Soft criteria (comma-separated):").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,0))
        self.e_soft = tk.Entry(conf_frame, width=60)
        self.e_soft.grid(row=1, column=2, columnspan=4, sticky="w", pady=(6,0))
        self.e_soft.insert(0, ",".join(DEFAULT_SOFT_CRITERIA))

        # Галка доверия (всегда включена и disabled)
        self.trust_var = tk.IntVar(value=1)
        self.chk_trust = tk.Checkbutton(conf_frame, text="Полностью довериться машине (по-умолчанию, недоступно)", variable=self.trust_var)
        self.chk_trust.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6,0))
        self.chk_trust.configure(state="disabled")

        # Run button
        run_btn = tk.Button(conf_frame, text="Рассчитать оптимальную конфигурацию", command=self._run_analysis)
        run_btn.grid(row=2, column=3, columnspan=2, sticky="w", padx=(6,0), pady=(6,0))

        # Output area: краткий и скрываемый подробный
        out_frame = tk.LabelFrame(self, text="Результат")
        out_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.summary_text = tk.Text(out_frame, height=10)
        self.summary_text.pack(fill="both", expand=False)

        # Toggle for details
        self.details_shown = False
        self.details_btn = tk.Button(out_frame, text="Показать детали (CI/CR, матрицы, веса)", command=self._toggle_details)
        self.details_btn.pack(pady=(6,0))
        self.details_panel = tk.Text(out_frame, height=12)
        # по умолчанию скрыт

        # ensure optcore2 path exists for reports
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optcore2")
        if not os.path.isdir(self.reports_dir):
            # create directory but do not create package — предполагается, что optcore2 уже создан
            try:
                os.makedirs(self.reports_dir, exist_ok=True)
            except Exception:
                pass

    # ------------------------
    # UI: Config management
    # ------------------------
    def _add_config_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Новая конфигурация")
        tk.Label(popup, text="ID конфигурации").pack()
        id_entry = tk.Entry(popup); id_entry.pack()
        tk.Label(popup, text="People (число людей)").pack()
        people_entry = tk.Entry(popup); people_entry.pack(); people_entry.insert(0, "1")

        def save():
            cid = id_entry.get().strip()
            if not cid:
                messagebox.showerror("Error", "ID не может быть пустым")
                return
            people = int(people_entry.get() or 0)
            self.configurations[cid] = {'devices': [], 'meta': {'people': people}}
            self.cfg_table.insert("", "end", iid=cid, values=(cid, people, 0))
            popup.destroy()
        tk.Button(popup, text="Сохранить", command=save).pack()

    def _delete_selected_config(self):
        sel = self.cfg_table.selection()
        if not sel:
            return
        for iid in sel:
            if iid in self.configurations:
                del self.configurations[iid]
            self.cfg_table.delete(iid)
        self.dev_table.delete(*self.dev_table.get_children())

    def _on_select_config(self, event):
        sel = self.cfg_table.selection()
        if not sel:
            return
        iid = sel[0]
        cfg = self.configurations.get(iid, {'devices': [], 'meta': {}})
        # populate dev_table
        self.dev_table.delete(*self.dev_table.get_children())
        for idx, d in enumerate(cfg['devices']):
            vals = (
                d.get('role',''),
                d.get('vendor',''),
                d.get('cpu_score',0),
                d.get('ram_score',0),
                d.get('energy',0),
                d.get('cost',0),
                d.get('reliability',{}).get('low',''),
                d.get('reliability',{}).get('high',''),
            )
            self.dev_table.insert("", "end", iid=f"{iid}_dev_{idx}", values=vals)

    # ------------------------
    # UI: device management
    # ------------------------
    def _add_device_popup(self):
        sel = self.cfg_table.selection()
        if not sel:
            messagebox.showerror("Error", "Сначала выберите конфигурацию")
            return
        cfg_id = sel[0]
        popup = tk.Toplevel(self)
        popup.title("Добавить устройство")

        entries = {}
        for label, default in [
            ("role", "client"), ("vendor",""), ("cpu_score","0.0"), ("ram_score","0.0"),
            ("energy","0.0"), ("cost","0.0"), ("rel_low","0.5"), ("rel_high","0.9")
        ]:
            tk.Label(popup, text=label).pack(anchor="w")
            e = tk.Entry(popup)
            e.pack(fill="x")
            e.insert(0, default)
            entries[label] = e

        def save_dev():
            d = {
                'role': entries['role'].get(),
                'vendor': entries['vendor'].get(),
                'cpu_score': float(entries['cpu_score'].get() or 0.0),
                'ram_score': float(entries['ram_score'].get() or 0.0),
                'energy': float(entries['energy'].get() or 0.0),
                'cost': float(entries['cost'].get() or 0.0),
                'reliability': {
                    'low': float(entries['rel_low'].get() or 0.0),
                    'high': float(entries['rel_high'].get() or 0.0)
                }
            }
            self.configurations[cfg_id]['devices'].append(d)
            # update table counts
            cnt = len(self.configurations[cfg_id]['devices'])
            people = self.configurations[cfg_id]['meta'].get('people', 0)
            self.cfg_table.item(cfg_id, values=(cfg_id, people, cnt))
            self._on_select_config(None)
            popup.destroy()

        tk.Button(popup, text="Добавить", command=save_dev).pack()

    def _delete_device(self):
        sel = self.dev_table.selection()
        if not sel:
            return
        item = sel[0]
        # parse id and index
        parts = item.split("_dev_")
        if len(parts) != 2:
            return
        cfg_id, idx = parts[0], int(parts[1])
        if cfg_id in self.configurations and idx < len(self.configurations[cfg_id]['devices']):
            del self.configurations[cfg_id]['devices'][idx]
            # rebuild dev table
            self._on_select_config(None)
            # update devices_count
            cnt = len(self.configurations[cfg_id]['devices'])
            people = self.configurations[cfg_id]['meta'].get('people', 0)
            self.cfg_table.item(cfg_id, values=(cfg_id, people, cnt))

    # ------------------------
    # JSON load/save
    # ------------------------
    def _load_from_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfgs = data.get("configurations", data if isinstance(data, list) else [])
            # clear current
            self.configurations.clear()
            for item in self.cfg_table.get_children():
                self.cfg_table.delete(item)
            # load
            for c in cfgs:
                cid = c.get('id') or c.get('name') or f"cfg_{len(self.configurations)+1}"
                devices = c.get('devices', [])
                people = int(c.get('meta', {}).get('people', c.get('people', 0)))
                self.configurations[cid] = {'devices': devices, 'meta': {'people': people}}
                self.cfg_table.insert("", "end", iid=cid, values=(cid, people, len(devices)))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить JSON: {e}")

    def _save_configs_to_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        out = {'configurations': []}
        for cid, v in self.configurations.items():
            out['configurations'].append({
                'id': cid,
                'devices': v['devices'],
                'meta': v.get('meta', {})
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("OK", "Сохранено")

    # ------------------------
    # Run AHP
    # ------------------------
    def _run_analysis(self):
        if run_ahp_pipeline is None:
            messagebox.showerror("Ошибка", "AHP engine не доступен. Проверьте optcore2/ahp_config_selector.py и __init__.py")
            return

        # Build configs list in expected format
        configs = []
        for cid, v in self.configurations.items():
            configs.append({
                'id': cid,
                'devices': v['devices'],
                'meta': v.get('meta', {})
            })

        # constraints from UI
        try:
            constraints = {
                'max_budget': float(self.e_budget.get()),
                'max_energy': float(self.e_energy.get()),
                'people_match_tolerance': float(self.e_tol.get())
            }
        except Exception as e:
            messagebox.showerror("Ошибка", f"Параметры constraints неверны: {e}")
            return

        soft_criteria = [s.strip() for s in self.e_soft.get().split(",") if s.strip()]
        if len(soft_criteria) == 0:
            soft_criteria = DEFAULT_SOFT_CRITERIA

        # Use embedded expert matrices (M1,M2)
        experts = [M1, M2]

        # call pipeline
        try:
            report = run_ahp_pipeline(
                configurations=configs,
                soft_criteria=soft_criteria,
                experts_criteria_matrices=experts,
                constraints={
                    'max_budget': constraints['max_budget'],
                    'max_energy': constraints['max_energy'],
                    'people_match_tolerance': constraints['people_match_tolerance']
                },
                saaty_cap=True,
                top_pct=0.10
            )
        except Exception as e:
            messagebox.showerror("Ошибка при выполнении AHP", str(e))
            return

        # Save report to optcore2/ahp_report_sample.json
        out_path = os.path.join(self.reports_dir, "ahp_report_sample.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showwarning("Warning", f"Не удалось сохранить отчёт: {e}")

        # Display brief summary
        self._display_summary(report)

        # fill details panel but keep hidden
        self._fill_details(report)

    def _display_summary(self, report):
        self.summary_text.delete("1.0", "end")
        lines = []
        lines.append(f"Всего входных конфигураций: {report.get('total_input')}")
        lines.append(f"Прошло фильтр: {report.get('passed_count')} (убрано: {report.get('removed_count')})")
        if 'warning' in report:
            lines.append("WARNING: " + report['warning'])
        crit = report.get('criteria', {})
        if crit:
            w = crit.get('weights', [])
            names = crit.get('names', [])
            lines.append("Критерии (веса):")
            for n, wt in zip(names, w):
                lines.append(f"  {n}: {wt:.4f}")
            lines.append(f"CR(criteria): {crit.get('CR'):.4f}  CI: {crit.get('CI'):.6f}")
        final = report.get('final', {})
        if final:
            lines.append("Рейтинг (top -> ...):")
            for aid, sc in final.get('ranking', []):
                lines.append(f"  {aid}: {sc:.4f}")
            sel = report.get('selection', {}).get('selected_ids', [])
            lines.append("Выбранные (top): " + ", ".join(sel))
        self.summary_text.insert("1.0", "\n".join(lines))

    def _fill_details(self, report):
        self.details_panel.delete("1.0", "end")
        pretty = json.dumps(report, ensure_ascii=False, indent=2)
        self.details_panel.insert("1.0", pretty)

    def _toggle_details(self):
        if self.details_shown:
            self.details_panel.pack_forget()
            self.details_shown = False
            self.details_btn.configure(text="Показать детали (CI/CR, матрицы, веса)")
        else:
            self.details_panel.pack(fill="both", expand=True)
            self.details_shown = True
            self.details_btn.configure(text="Скрыть детали")

