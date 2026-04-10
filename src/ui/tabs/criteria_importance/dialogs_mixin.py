import tkinter as tk
from tkinter import ttk, messagebox


class CriteriaImportanceDialogsMixin:
    def _criterion_popup(self, mode="add", criterion=None):
        if not self.case_data:
            return
        criterion = criterion or {"id": self._suggest_new_criterion_id(), "name": "Новый критерий"}

        popup = tk.Toplevel(self)
        popup.title("Критерий")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        id_var = tk.StringVar(value=criterion["id"])
        name_var = tk.StringVar(value=criterion["name"])
        default_var = tk.StringVar(value="3")

        form = tk.Frame(popup)
        form.pack(padx=10, pady=10)
        tk.Label(form, text="ID критерия").grid(row=0, column=0, sticky="w")
        tk.Entry(form, textvariable=id_var, width=22).grid(row=0, column=1, sticky="w")
        tk.Label(form, text="Название").grid(row=1, column=0, sticky="w")
        tk.Entry(form, textvariable=name_var, width=42).grid(row=1, column=1, sticky="w")
        if mode == "add":
            tk.Label(form, text="Стартовая оценка для всех систем").grid(
                row=2, column=0, sticky="w"
            )
            tk.Entry(form, textvariable=default_var, width=12).grid(row=2, column=1, sticky="w")

        def save():
            new_id = id_var.get().strip()
            new_name = name_var.get().strip()
            if not new_id or not new_name:
                messagebox.showerror("Ошибка", "Заполните ID и название критерия")
                return
            old_id = criterion["id"]
            if new_id != old_id and new_id in self._criterion_ids():
                messagebox.showerror("Ошибка", "Критерий с таким ID уже существует")
                return
            if mode == "add":
                try:
                    default_score = float(default_var.get().replace(",", "."))
                except Exception:
                    messagebox.showerror("Ошибка", "Стартовая оценка должна быть числом")
                    return
                self.case_data["criteria"].append({"id": new_id, "name": new_name})
                for alt_id in self._alternative_ids():
                    self.case_data["scores"].setdefault(alt_id, {})[new_id] = default_score
            else:
                for item in self.case_data["criteria"]:
                    if item["id"] == old_id:
                        item["id"] = new_id
                        item["name"] = new_name
                        break
                for alt_id in self._alternative_ids():
                    self.case_data["scores"][alt_id][new_id] = self.case_data["scores"][alt_id].pop(
                        old_id
                    )
                for rel in self.case_data.get("relations", []):
                    if rel["left"] == old_id:
                        rel["left"] = new_id
                    if rel["right"] == old_id:
                        rel["right"] = new_id
            self.analysis_report = None
            self._refresh_scores_tree()
            self._refresh_relations_tree()
            popup.destroy()

        tk.Button(popup, text="Сохранить", command=save).pack(padx=10, pady=(0, 10))

    def _add_criterion_popup(self):
        self._criterion_popup(mode="add")

    def _edit_selected_criterion(self):
        criterion_id = self._selected_criterion_id()
        if not criterion_id:
            messagebox.showinfo("Критерий", "Сначала выделите строку критерия в первой таблице")
            return
        criterion = next((item for item in self._criteria() if item["id"] == criterion_id), None)
        if not criterion:
            return
        self._criterion_popup(mode="edit", criterion=dict(criterion))

    def _delete_selected_criterion(self):
        criterion_id = self._selected_criterion_id()
        if not criterion_id:
            messagebox.showinfo("Критерий", "Сначала выделите строку критерия в первой таблице")
            return
        if not messagebox.askyesno("Удаление критерия", f"Удалить критерий {criterion_id}?"):
            return
        self.case_data["criteria"] = [c for c in self._criteria() if c["id"] != criterion_id]
        for alt_id in self._alternative_ids():
            self.case_data["scores"].get(alt_id, {}).pop(criterion_id, None)
        self.case_data["relations"] = [
            rel
            for rel in self.case_data.get("relations", [])
            if rel["left"] != criterion_id and rel["right"] != criterion_id
        ]
        self.analysis_report = None
        self._refresh_scores_tree()
        self._refresh_relations_tree()

    def _alternative_popup(self, mode="add", alt=None):
        if not self.case_data:
            return
        alt = alt or {
            "id": self._suggest_new_alternative_id(),
            "name": f"Система {len(self._alternatives()) + 1}",
        }

        popup = tk.Toplevel(self)
        popup.title("Система")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        id_var = tk.StringVar(value=alt["id"])
        name_var = tk.StringVar(value=alt["name"])
        default_var = tk.StringVar(value="3")

        form = tk.Frame(popup)
        form.pack(padx=10, pady=10)
        tk.Label(form, text="ID системы").grid(row=0, column=0, sticky="w")
        tk.Entry(form, textvariable=id_var, width=24).grid(row=0, column=1, sticky="w")
        tk.Label(form, text="Название").grid(row=1, column=0, sticky="w")
        tk.Entry(form, textvariable=name_var, width=36).grid(row=1, column=1, sticky="w")
        if mode == "add":
            tk.Label(form, text="Стартовая оценка по всем критериям").grid(
                row=2, column=0, sticky="w"
            )
            tk.Entry(form, textvariable=default_var, width=12).grid(row=2, column=1, sticky="w")

        def save():
            new_id = id_var.get().strip()
            new_name = name_var.get().strip()
            if not new_id or not new_name:
                messagebox.showerror("Ошибка", "Заполните ID и название системы")
                return
            old_id = alt["id"]
            if new_id != old_id and new_id in self._alternative_ids():
                messagebox.showerror("Ошибка", "Система с таким ID уже существует")
                return
            if mode == "add":
                try:
                    default_score = float(default_var.get().replace(",", "."))
                except Exception:
                    messagebox.showerror("Ошибка", "Стартовая оценка должна быть числом")
                    return
                self.case_data["alternatives"].append({"id": new_id, "name": new_name})
                self.case_data["scores"][new_id] = {
                    criterion_id: default_score for criterion_id in self._criterion_ids()
                }
            else:
                for item in self.case_data["alternatives"]:
                    if item["id"] == old_id:
                        item["id"] = new_id
                        item["name"] = new_name
                        break
                self.case_data["scores"][new_id] = self.case_data["scores"].pop(old_id)
            self.analysis_report = None
            self._refresh_scores_tree()
            popup.destroy()

        tk.Button(popup, text="Сохранить", command=save).pack(padx=10, pady=(0, 10))

    def _choose_alternative_popup(self, title, action):
        if not self.case_data or not self._alternatives():
            return
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        selected_var = tk.StringVar(value=self._alternative_ids()[0])
        tk.Label(popup, text="Выберите систему").pack(padx=10, pady=(10, 4))
        ttk.Combobox(
            popup,
            textvariable=selected_var,
            values=[f"{alt['id']} — {alt['name']}" for alt in self._alternatives()],
            state="readonly",
            width=40,
        ).pack_forget()
        # отдельный combobox с id, чтобы не усложнять парсинг
        box = ttk.Combobox(
            popup,
            textvariable=selected_var,
            values=self._alternative_ids(),
            state="readonly",
            width=20,
        )
        box.pack(padx=10, pady=4)

        def run_action():
            action(selected_var.get())
            popup.destroy()

        tk.Button(popup, text="Продолжить", command=run_action).pack(padx=10, pady=(4, 10))

    def _add_alternative_popup(self):
        self._alternative_popup(mode="add")

    def _edit_alternative_popup(self):
        def edit(alt_id):
            alt = next((item for item in self._alternatives() if item["id"] == alt_id), None)
            if alt:
                self._alternative_popup(mode="edit", alt=dict(alt))

        self._choose_alternative_popup("Изменить систему", edit)

    def _delete_alternative_popup(self):
        def delete(alt_id):
            if len(self._alternatives()) <= 2:
                messagebox.showerror("Ошибка", "Для анализа нужны минимум две системы")
                return
            if not messagebox.askyesno(
                "Удаление системы", f"Удалить систему {self._alternative_name(alt_id)}?"
            ):
                return
            self.case_data["alternatives"] = [a for a in self._alternatives() if a["id"] != alt_id]
            self.case_data["scores"].pop(alt_id, None)
            self.analysis_report = None
            self._refresh_scores_tree()

        self._choose_alternative_popup("Удалить систему", delete)

    def _relation_popup(self, initial=None, index=None):
        if not self.case_data or not self._criteria():
            return
        first_id = self._criterion_ids()[0]
        initial = initial or {"left": first_id, "op": "=", "factor": 1.0, "right": first_id}
        popup = tk.Toplevel(self)
        popup.title("Связь важности критериев")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        ids = self._criterion_ids()
        left_var = tk.StringVar(value=initial["left"])
        op_var = tk.StringVar(value=initial.get("op", "="))
        factor_var = tk.StringVar(value=str(initial.get("factor", 1.0)))
        right_var = tk.StringVar(value=initial["right"])

        form = tk.Frame(popup)
        form.pack(padx=10, pady=10)

        tk.Label(form, text="Левый критерий").grid(row=0, column=0, sticky="w")
        ttk.Combobox(form, textvariable=left_var, values=ids, state="readonly", width=12).grid(
            row=0, column=1, sticky="w"
        )
        tk.Label(form, text="Операция").grid(row=1, column=0, sticky="w")
        ttk.Combobox(form, textvariable=op_var, values=["=", ">"], state="readonly", width=12).grid(
            row=1, column=1, sticky="w"
        )
        tk.Label(form, text="Коэффициент").grid(row=2, column=0, sticky="w")
        tk.Entry(form, textvariable=factor_var, width=14).grid(row=2, column=1, sticky="w")
        tk.Label(form, text="Правый критерий").grid(row=3, column=0, sticky="w")
        ttk.Combobox(form, textvariable=right_var, values=ids, state="readonly", width=12).grid(
            row=3, column=1, sticky="w"
        )

        def save_relation():
            try:
                relation = {
                    "left": left_var.get(),
                    "op": op_var.get(),
                    "factor": float(factor_var.get().replace(",", ".")),
                    "right": right_var.get(),
                }
            except Exception:
                messagebox.showerror("Ошибка", "Некорректный коэффициент")
                return
            if index is None:
                self.case_data.setdefault("relations", []).append(relation)
            else:
                self.case_data["relations"][index] = relation
            self.analysis_report = None
            self._refresh_relations_tree()
            popup.destroy()

        tk.Button(popup, text="Сохранить", command=save_relation).pack(padx=10, pady=(0, 10))

    def _add_relation_popup(self):
        self._relation_popup()

    def _edit_relation_popup(self, _event=None):
        selected = self.relations_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        self._relation_popup(initial=self.case_data["relations"][idx], index=idx)

    def _delete_relation(self):
        selected = self.relations_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        del self.case_data["relations"][idx]
        self.analysis_report = None
        self._refresh_relations_tree()
